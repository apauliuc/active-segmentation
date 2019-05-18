import abc
import os
import random
import numpy as np
from typing import Tuple
from logging import Logger

import torch
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from ignite import handlers, metrics
from ignite.engine.engine import Engine, Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator

from data.base_loader import BaseLoader
from models import get_model
from losses import get_loss_function, BCEAndJaccardLoss
from optimizers import get_optimizer
from helpers.config import ConfigClass
from helpers.metrics import SegmentationMetrics
from helpers.utils import setup_logger, retrieve_class_init_parameters, timer_to_str
from helpers.paths import get_resume_model_path, get_resume_optimizer_path


class BaseTrainer(abc.ABC):
    data_loaders: BaseLoader
    len_models: int

    def __init__(self, config: ConfigClass, save_dir: str, log_name=''):
        self.save_dir = save_dir
        self.main_logger, self.main_log_handler = setup_logger(save_dir, log_name)
        self.main_logger.info(f'Saving to folder {save_dir}')
        self.main_writer = SummaryWriter(log_dir=save_dir)

        self.config = config
        self.model_cfg = config.model
        self.train_cfg = config.training
        self.optim_cfg = config.training.optimizer
        self.resume_cfg = config.resume

        if self.train_cfg.seed is not None:
            torch.manual_seed(self.train_cfg.seed)
            random.seed(self.train_cfg.seed)
            np.random.seed(self.train_cfg.seed)
            self.main_logger.info(f'Seed set on {self.train_cfg.seed}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_train_loader = config.data.run_val_on_train

        if self.train_cfg.early_stop_fn == 'f1_score':
            self.eval_func = self.f1_score
        elif self.train_cfg.early_stop_fn == 'iou_score':
            self.eval_func = self.iou_score
        else:
            self.eval_func = self.val_loss

        self.use_ensemble = self.train_cfg.use_ensemble
        if self.train_cfg.use_ensemble:
            self.len_models = self.train_cfg.ensemble.number_models

    # Single model
    def _init_train_components(self):
        self.metrics = {
            'loss': metrics.Loss(get_loss_function(self.train_cfg.loss_fn)),
            'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                   threshold=self.config.binarize_threshold)
        }

        self.model_cfg.network_params.input_channels = self.data_loaders.input_channels
        self.model_cfg.network_params.num_classes = self.data_loaders.num_classes

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()

        self.lr_scheduler = self._init_lr_scheduler()

        self.trainer, self.evaluator = self._init_engines()

        self._init_handlers()

    def _init_model(self):
        model = get_model(self.model_cfg).to(device=self.device)
        self.main_logger.info(f'Using model {model}')

        if self.resume_cfg.resume_from is not None and self.resume_cfg.saved_model is not None:
            model_path = get_resume_model_path(self.resume_cfg.resume_from, self.resume_cfg.saved_model)
            self.main_logger.info(f'Loading model loaded from {model_path}')
            model.load_state_dict(torch.load(model_path))

        return model

    def _init_optimizer(self):
        optimizer_cls = get_optimizer(self.optim_cfg)

        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in self.optim_cfg.items() if k in init_param_names}

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.main_logger.info(f'Using optimizer {optimizer.__class__.__name__}')

        if self.resume_cfg.resume_from is not None and self.resume_cfg.saved_optimizer is not None:
            optimizer_path = get_resume_optimizer_path(self.resume_cfg.resume_from, self.resume_cfg.saved_optimizer)
            self.main_logger.info(f'Loading optimizer from {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))

        return optimizer

    def _init_criterion(self):
        criterion = get_loss_function(self.train_cfg.loss_fn).to(device=self.device)
        self.main_logger.info(f'Using loss function {criterion}')

        return criterion

    def _init_lr_scheduler(self):
        lr_scheduler = None
        if self.optim_cfg.scheduler == 'step':
            lr_scheduler = StepLR(self.optimizer, step_size=self.optim_cfg.lr_cycle, gamma=0.1)

        return lr_scheduler

    def _init_engines(self) -> Tuple[Engine, Engine]:
        if self.use_ensemble:
            trainer = self._init_trainer_engine_ensemble()
            evaluator = self._init_evaluator_engine_ensemble()
        else:
            trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)
            evaluator = create_supervised_evaluator(self.model, self.metrics, self.device, True)

        metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')

        return trainer, evaluator

    # Ensemble Method
    def _init_train_components_ensemble(self):
        self.metrics = {
            'loss': metrics.Loss(BCEAndJaccardLoss(ensemble=True)),
            'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                   threshold=self.config.binarize_threshold,
                                                   ensemble=True)
        }

        self.model_cfg.network_params.input_channels = self.data_loaders.input_channels
        self.model_cfg.network_params.num_classes = self.data_loaders.num_classes

        self.ens_models = list()
        self.ens_optimizers = list()
        self.ens_lr_schedulers = list()

        optimizer_cls = get_optimizer(self.optim_cfg)
        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in self.optim_cfg.items() if k in init_param_names}

        for _ in range(self.len_models):
            self.ens_models.append(get_model(self.model_cfg).to(device=self.device))
            self.ens_optimizers.append(optimizer_cls(self.ens_models[-1].parameters(), **optimizer_params))

            lr_scheduler = None
            if self.optim_cfg.scheduler == 'step':
                lr_scheduler = StepLR(self.ens_optimizers[-1], step_size=self.optim_cfg.lr_cycle, gamma=0.1)

            self.ens_lr_schedulers.append(lr_scheduler)

        self.main_logger.info(f'Using ensemble of {self.len_models} {self.ens_models[0]}')
        self.main_logger.info(f'Using optimizers {self.ens_optimizers[0].__class__.__name__}')

        self.criterion = self._init_criterion()

        self.trainer, self.evaluator = self._init_engines()

        self._init_handlers()

    def _init_trainer_engine_ensemble(self) -> Engine:
        # noinspection PyUnusedLocal
        def _update(engine, batch):
            for model, optimizer in zip(self.ens_models, self.ens_optimizers):
                model.train()
                optimizer.zero_grad()

            x, y = batch
            x = x.to(device=self.device, non_blocking=True)
            y = y.to(device=self.device, non_blocking=True)

            avg_loss = torch.zeros(1)
            for model, optimizer in zip(self.ens_models, self.ens_optimizers):
                y_pred = model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

            return avg_loss.item() / self.len_models

        return Engine(_update)

    def _init_evaluator_engine_ensemble(self) -> Engine:
        # noinspection PyUnusedLocal
        def _inference(engine_, batch):
            for model in self.ens_models:
                model.eval()

            with torch.no_grad():
                x, y = batch
                x = x.to(device=self.device, non_blocking=True)
                y = y.to(device=self.device, non_blocking=True)

                avg_pred = torch.zeros_like(y)
                for model in self.ens_models:
                    y_pred = torch.sigmoid(model(x))
                    avg_pred += y_pred

                avg_pred /= self.len_models

                return avg_pred, y

        engine = Engine(_inference)

        for name, metric in self.metrics.items():
            metric.attach(engine, name)

        return engine

    # General components
    def _init_epoch_timer(self) -> None:
        self.timer = handlers.Timer(average=False)
        self.timer.attach(self.trainer,
                          start=Events.EPOCH_STARTED,
                          resume=Events.ITERATION_STARTED,
                          pause=Events.ITERATION_COMPLETED)

    def _init_checkpoint_handler(self, save_dir=None) -> None:
        save_dir = self.save_dir if save_dir is None else save_dir

        if self.use_ensemble:
            checkpoint_save = {f'model_{i}': model for i, model in enumerate(self.ens_models)}
        else:
            checkpoint_save = {'model': self.model, 'optimizer': self.optimizer}

        best_ckpoint_handler = handlers.ModelCheckpoint(save_dir, 'best', n_saved=1, require_empty=False,
                                                        score_function=self.eval_func, save_as_state_dict=True)

        final_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpoint_handler, checkpoint_save)
        self.trainer.add_event_handler(Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def _init_early_stopping_handler(self) -> None:
        early_stop_handler = handlers.EarlyStopping(self.train_cfg.patience,
                                                    score_function=self.eval_func,
                                                    trainer=self.trainer)
        self.evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    def _on_epoch_started(self, _engine: Engine) -> None:
        if self.use_ensemble:
            for lr_scheduler in self.ens_lr_schedulers:
                if lr_scheduler is not None:
                    lr_scheduler.step(_engine.state.epoch)
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: Engine) -> None:
        self._log_training_results(_engine, self.main_logger, self.main_writer)
        self._evaluate_on_val(_engine, self.main_logger, self.main_writer)

    def _on_events_completed(self, _engine: Engine) -> None:
        self._finalize()

    def _on_exception_raised(self, _engine: Engine, e: Exception) -> None:
        self.main_logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.main_logger.info(e)
        self._finalize()
        raise e

    def _finalize(self) -> None:
        if self.trainer.should_terminate:
            self.main_logger.info(f'Early stopping on epoch {self.trainer.state.epoch}')

        self.main_writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.main_writer.close()
        self.main_logger.removeHandler(self.main_log_handler)

    def _log_training_results(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        train_duration = timer_to_str(self.timer.value())
        avg_loss = _train_engine.state.metrics['train_loss']
        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}. ' \
            f'Duration: {train_duration}. Avg loss: {avg_loss:.4f}'
        logger.info(msg)
        writer.add_scalar('training/avg_loss', avg_loss, _train_engine.state.epoch)

    def _evaluate_on_val(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        self.evaluator.run(self.data_loaders.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']
        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}   ' \
            f'IoU: {eval_metrics["avg_iou"]:.4f}   ' \
            f'F1: {eval_metrics["avg_f1"]:.4f}'
        logger.info(msg)
        writer.add_scalar('validation_eval/avg_loss', eval_loss, _train_engine.state.epoch)

        for key, value in eval_metrics.items():
            writer.add_scalar(f'val_metrics/{key}', value, _train_engine.state.epoch)

    @staticmethod
    def val_loss(_engine: Engine) -> float:
        return -round(_engine.state.metrics['loss'], 6)

    @staticmethod
    def iou_score(_engine: Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_iou'], 6)

    @staticmethod
    def f1_score(_engine: Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_f1'], 6)

    @abc.abstractmethod
    def _init_handlers(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass
