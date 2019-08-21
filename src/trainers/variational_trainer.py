import os
from typing import Tuple
from logging import Logger
from torch.utils.tensorboard import SummaryWriter

import torch
from ignite import metrics
from ignite.engine.engine import Engine

from data import get_dataloaders
from helpers.config import ConfigClass
from helpers.metrics import VAEMetrics, SegmentationMetrics
from helpers.utils import timer_to_str
from losses.vae_criterion import VAECriterion
from trainers.base_trainer import BaseTrainer
from definitions import RUNS_DIR


class VariationalTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str, log_name='Variational_Trainer'):
        super(VariationalTrainer, self).__init__(config, save_dir, log_name)

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self.starting_kld_factor = 0 if self.loss_cfg.kld_warmup else self.loss_cfg.kld_factor
        self.starting_mse_factor = 0 if self.loss_cfg.mse_warmup else self.loss_cfg.mse_factor

        self._init_train_components()

    def _init_train_components(self, reinitialise=False):
        super()._init_train_components()

        self.vae_criterion = VAECriterion(ce_loss=self.criterion,
                                          mse_factor=self.starting_mse_factor,
                                          kld_factor=self.starting_kld_factor,
                                          prior_var=self.loss_cfg.prior_var)
        self.main_logger.info(f'Final loss is {self.vae_criterion}')
        self.model.register_mean_std(self.data_loaders.ds_statistics, self.device)

        if self.model_cfg.network_params.load_standard_unet:
            self.main_logger.info('Loading weights from standard UNet')
            pretrained_dict = torch.load(os.path.join(RUNS_DIR, 'standard.pth'))
            model_dict = self.model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.model.to(self.device)

    def _init_engines(self) -> Tuple[Engine, Engine]:
        self.train_metrics = {
            'total_loss': metrics.RunningAverage(output_transform=lambda x: x['loss']),
            'segment_loss': metrics.RunningAverage(output_transform=lambda x: x['segment_loss']),
            'recon_loss': metrics.RunningAverage(output_transform=lambda x: x['recon_loss']),
            'kl_div': metrics.RunningAverage(output_transform=lambda x: x['kl_div'])
        }

        self.val_metrics = {
            'vae_metrics': VAEMetrics(loss_fn=self.criterion,
                                      mse_factor=self.starting_mse_factor,
                                      kld_factor=self.starting_kld_factor),
            'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                   threshold=self.config.binarize_threshold),
        }

        trainer = self._init_trainer_engine()
        evaluator = self._init_evaluator_engine()

        return trainer, evaluator

    def _init_trainer_engine(self) -> Engine:
        self.model.to(self.device)

        def _update(_engine, batch):
            self.model.train()

            x, y = batch
            x = x.to(device=self.device, non_blocking=True)
            y = y.to(device=self.device, non_blocking=True)

            pred, recon, mu, var = self.model(x)

            loss, ce, mse, kl_div = self.vae_criterion(pred, y, recon, x, mu, var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return {
                'loss': loss.item(),
                'segment_loss': ce.item(),
                'recon_loss': mse.item(),
                'kl_div': kl_div.item()
            }

        _engine = Engine(_update)

        if self.train_metrics is not None:
            for name, metric in self.train_metrics.items():
                metric.attach(_engine, name)

        return _engine

    def _init_evaluator_engine(self) -> Engine:
        self.model.to(self.device)

        def _inference(_engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch
                x = x.to(device=self.device, non_blocking=True)
                y = y.to(device=self.device, non_blocking=True)

                pred, recon, mu, var = self.model(x)

                return pred, y, recon, x, mu, var

        _engine = Engine(_inference)

        if self.val_metrics is not None:
            for name, metric in self.val_metrics.items():
                metric.attach(_engine, name)

        return _engine

    def _on_epoch_started(self, _engine: Engine) -> None:
        super()._on_epoch_started(_engine)
        if self.loss_cfg.mse_warmup:
            self._step_mse(_engine)
        if self.loss_cfg.kld_warmup:
            self._step_kld(_engine)

    def _log_training_results(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        train_duration = timer_to_str(self.timer.value())
        _metrics = _train_engine.state.metrics

        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}  ' \
              f'Duration: {train_duration} | ' \
              f'Avg loss: {_metrics["total_loss"]:.4f}  ||  ' \
              f'Seg loss: {_metrics["segment_loss"]:.4f} | ' \
              f'Recon loss: {_metrics["recon_loss"]:.4f} | ' \
              f'KL Div: {_metrics["kl_div"]:.4f}'
        logger.info(msg)

        for name, value in _train_engine.state.metrics.items():
            writer.add_scalar(f'training/{name}', value, _train_engine.state.epoch)

    def _evaluate_on_val(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        self.evaluator.run(self.data_loaders.val_loader)

        vae_metrics = self.evaluator.state.metrics['vae_metrics']
        self.evaluator.state.metrics['loss'] = vae_metrics['total_loss']
        segment_metrics = self.evaluator.state.metrics['segment_metrics']

        msg = f'Eval. on val_loader - Avg loss: {vae_metrics["total_loss"]:.4f}    ||     ' \
              f'Seg loss: {vae_metrics["segment_loss"]:.4f} | ' \
              f'Recon loss: {vae_metrics["recon_loss"]:.4f} | ' \
              f'KL Div: {vae_metrics["kl_div"]:.4f}     ||     ' \
              f'IoU: {segment_metrics["avg_iou"]:.4f} | F1: {segment_metrics["avg_f1"]:.4f}'
        logger.info(msg)

        for name, value in vae_metrics.items():
            writer.add_scalar(f'validation/{name}', value, _train_engine.state.epoch)

        for name, value in segment_metrics.items():
            writer.add_scalar(f'validation_segment_metrics/{name}', value, _train_engine.state.epoch)

    @staticmethod
    def _step_factor(_epoch, _current, _config_factor, _type, _step_interval, _step_size) -> Tuple[float, bool]:
        if _epoch != 0:
            if _type == 'interval' and _epoch % _step_interval == 0 and _current < _config_factor:
                step = _epoch // _step_interval
                return min(_config_factor, step * _step_size), True
            elif _type == 'gradual' and _epoch >= _step_interval and _current < _config_factor:
                step = _epoch + 1 - _step_interval
                return min(_config_factor, step * _step_size), True
        return _current, False

    def _step_kld(self, _engine: Engine) -> None:
        """Warm-up method for KL Divergence"""
        kld_factor, updated = self._step_factor(_epoch=_engine.state.epoch,
                                                _current=self.vae_criterion.kld_factor,
                                                _config_factor=self.loss_cfg.kld_factor,
                                                _type=self.loss_cfg.kld_factor_type,
                                                _step_interval=self.loss_cfg.kld_step_interval,
                                                _step_size=self.loss_cfg.kld_step_size)

        if updated:
            self.vae_criterion.kld_factor = kld_factor
            self.val_metrics['vae_metrics'].update_kld_factor(kld_factor)
            self.main_logger.info(f'KLD factor changed to {kld_factor} '
                                  f'at epoch {_engine.state.epoch}')

    def _step_mse(self, _engine):
        """Warm-up method for MSE Loss"""
        mse_factor, updated = self._step_factor(_epoch=_engine.state.epoch,
                                                _current=self.vae_criterion.mse_factor,
                                                _config_factor=self.loss_cfg.mse_factor,
                                                _type=self.loss_cfg.mse_factor_type,
                                                _step_interval=self.loss_cfg.mse_step_interval,
                                                _step_size=self.loss_cfg.mse_step_size)
        if updated:
            self.vae_criterion.mse_factor = mse_factor
            self.val_metrics['vae_metrics'].update_mse_factor(mse_factor)
            self.main_logger.info(f'MSE factor changed to {mse_factor} '
                                  f'at epoch {_engine.state.epoch}')

    def run(self) -> None:
        self.main_logger.info(f'Current MSE factor = {self.vae_criterion.mse_factor} | '
                              f'Current KLD factor = {self.vae_criterion.kld_factor}')
        super().run()
