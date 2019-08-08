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


class VariationalTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(VariationalTrainer, self).__init__(config, save_dir, 'Variational_Trainer')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self.starting_kld_factor = 0 if self.train_cfg.loss_fn.kld_warmup else self.train_cfg.loss_fn.kld_factor

        self._init_train_components()

    def _init_train_components(self, reinitialise=False):
        super()._init_train_components()

        self.vae_criterion = VAECriterion(ce_loss=self.criterion,
                                          mse_factor=self.train_cfg.loss_fn.mse_factor,
                                          kld_factor=self.starting_kld_factor)
        self.model.register_mean_std(self.data_loaders.ds_statistics, self.device)

    def _init_engines(self) -> Tuple[Engine, Engine]:
        self.train_metrics = {
            'total_loss': metrics.RunningAverage(output_transform=lambda x: x['loss']),
            'seg_loss': metrics.RunningAverage(output_transform=lambda x: x['segment_loss']),
            'kl_div': metrics.RunningAverage(output_transform=lambda x: x['kl_div']),
            'recon_loss': metrics.RunningAverage(output_transform=lambda x: x['recon_loss'])
        }

        self.val_metrics = {
            'vae_metrics': VAEMetrics(loss_fn=self.criterion,
                                      mse_factor=self.train_cfg.loss_fn.mse_factor,
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
                'kl_div': kl_div.item(),
                'recon_loss': mse.item()
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

    def _on_epoch_completed(self, _engine: Engine) -> None:
        super()._on_epoch_completed(_engine)
        if self.train_cfg.loss_fn.kld_warmup:
            self._step_kld(_engine)

    def _log_training_results(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        train_duration = timer_to_str(self.timer.value())
        _metrics = _train_engine.state.metrics

        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}  ' \
            f'Duration: {train_duration}  ||  ' \
            f'Avg loss: {_metrics["total_loss"]:.4f} | ' \
            f'Seg loss: {_metrics["seg_loss"]:.4f} | ' \
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

    def _step_kld(self, _engine: Engine) -> None:
        """Warm-up method for KL Divergence"""
        if _engine.state.epoch != 0 and _engine.state.epoch % self.train_cfg.loss_fn.kld_step_interval == 0:
            step = (_engine.state.epoch + 1) // self.train_cfg.loss_fn.kld_step_interval
            self.vae_criterion.kld_factor = min(self.train_cfg.loss_fn.kld_factor,
                                                step * self.train_cfg.loss_fn.kld_step_size)
            self.val_metrics['vae_metrics'].update_kld_factor(self.vae_criterion.kld_factor)
            self.main_logger.info(f'KLD factor changed to {self.vae_criterion.kld_factor} '
                                  f'at epoch {_engine.state.epoch}')

    def run(self) -> None:
        self.main_logger.info(f'Initial KLD factor = {self.vae_criterion.kld_factor}')
        super().run()
