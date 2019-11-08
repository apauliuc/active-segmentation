import abc
import random
from logging import Logger
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from ignite import handlers, metrics
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.engine.engine import Engine, Events
from torch.utils.tensorboard import SummaryWriter

from data.base_loader import BaseLoader
from helpers.config import ConfigClass
from helpers.metrics import SegmentationMetrics
from helpers.paths import get_resume_model_path, get_resume_optimizer_path
from helpers.utils import setup_logger, retrieve_class_init_parameters, timer_to_str
from losses import get_loss_function, BCEAndJaccardLoss
from models import get_model
from optimizers import get_optimizer


class BaseTrainer(abc.ABC):
    data_loaders: BaseLoader
    len_models: int

    def __init__(self, config: ConfigClass, save_dir: str, log_name=''):
        self.log_name = f'{log_name}_{config.gpu_node}'
        self.save_dir = save_dir
        self.main_logger, self.main_log_handler = setup_logger(save_dir, self.log_name)
        self.main_logger.info(f'Saving to folder {save_dir}')
        self.main_writer = SummaryWriter(save_dir)

        self.config = config
        self.model_cfg = config.model
        self.train_cfg = config.training
        self.optim_cfg = config.training.optimizer
        self.loss_cfg = config.training.loss_fn
        self.resume_cfg = config.resume

        if self.train_cfg.seed is not None:
            torch.manual_seed(self.train_cfg.seed)
            random.seed(self.train_cfg.seed)
            np.random.seed(self.train_cfg.seed)
            self.main_logger.info(f'Seed set on {self.train_cfg.seed}')

        self.device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')
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

    def _init_model(self, init=True):
        model = get_model(self.model_cfg).to(device=self.device)
        if init:
            self.main_logger.info(f'Using model {model}')

        if self.resume_cfg.resume_from is not None and self.resume_cfg.saved_model is not None and init:
            model_path = get_resume_model_path(self.resume_cfg.resume_from, self.resume_cfg.saved_model)
            if init:
                self.main_logger.info(f'Loading model loaded from {model_path}')
            model.load_state_dict(torch.load(model_path))

        return model

    def _init_optimizer(self, init=True):
        optimizer_cls = get_optimizer(self.optim_cfg)

        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in self.optim_cfg.items() if k in init_param_names}

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        if init:
            self.main_logger.info(f'Using optimizer {optimizer.__class__.__name__}')

        if self.resume_cfg.resume_from is not None and self.resume_cfg.saved_optimizer is not None and init:
            optimizer_path = get_resume_optimizer_path(self.resume_cfg.resume_from, self.resume_cfg.saved_optimizer)
            if init:
                self.main_logger.info(f'Loading optimizer from {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))

        return optimizer

    def _init_criterion(self):
        criterion = get_loss_function(self.loss_cfg).to(device=self.device)
        self.main_logger.info(f'Using loss function {criterion}')

        return criterion

    def _init_lr_scheduler(self, optimizer):
        if self.optim_cfg.scheduler == 'step':
            scheduler_class = optim.lr_scheduler.StepLR
        elif self.optim_cfg.scheduler == 'plateau':
            scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
        else:
            return None

        init_param_names = retrieve_class_init_parameters(scheduler_class)
        scheduler_params = {k: v for k, v in self.optim_cfg.scheduler_params.items() if k in init_param_names}

        lr_scheduler = scheduler_class(optimizer, **scheduler_params)

        return lr_scheduler

    def _init_engines(self) -> Tuple[Engine, Engine]:
        if self.use_ensemble:
            trainer = self._init_trainer_engine_ensemble()
            evaluator = self._init_evaluator_engine_ensemble()
        else:
            trainer = self._init_trainer_engine()
            evaluator = self._init_evaluator_engine()

        metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')

        return trainer, evaluator

    # Single model
    def _init_train_components(self, reinitialise=False):
        if not reinitialise:
            self.val_metrics = {
                'loss': metrics.Loss(get_loss_function(self.loss_cfg)),
                'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                       threshold=self.config.binarize_threshold)
            }

            self.model_cfg.network_params.input_channels = self.data_loaders.input_channels
            self.model_cfg.network_params.num_classes = self.data_loaders.num_classes
            self.model_cfg.network_params.image_size = self.data_loaders.image_size

        self.model = self._init_model(not reinitialise)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(not reinitialise)

        self.lr_scheduler = self._init_lr_scheduler(self.optimizer)

        self.trainer, self.evaluator = self._init_engines()

        self._init_handlers()

    def _init_trainer_engine(self) -> Engine:
        return create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)

    def _init_evaluator_engine(self) -> Engine:
        return create_supervised_evaluator(self.model, self.val_metrics, self.device, True)

    # Ensemble Method
    def _init_train_components_ensemble(self, reinitialise=False):
        if not reinitialise:
            self.val_metrics = {
                'loss': metrics.Loss(BCEAndJaccardLoss(eval_ensemble=True, gpu_node=self.config.gpu_node)),
                'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                       threshold=self.config.binarize_threshold,
                                                       eval_ensemble=True)
            }

            self.model_cfg.network_params.input_channels = self.data_loaders.input_channels
            self.model_cfg.network_params.num_classes = self.data_loaders.num_classes
            self.model_cfg.network_params.image_size = self.data_loaders.image_size

        self.criterion = self._init_criterion()
        self.ens_models = list()
        self.ens_optimizers = list()
        self.ens_lr_schedulers = list()

        optimizer_cls = get_optimizer(self.optim_cfg)
        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in self.optim_cfg.items() if k in init_param_names}

        for _ in range(self.len_models):
            self.ens_models.append(get_model(self.model_cfg).to(device=self.device))
            self.ens_optimizers.append(optimizer_cls(self.ens_models[-1].parameters(), **optimizer_params))

            lr_scheduler = self._init_lr_scheduler(self.ens_optimizers[-1])
            self.ens_lr_schedulers.append(lr_scheduler)

        if not reinitialise:
            self.main_logger.info(f'Using ensemble of {self.len_models} {self.ens_models[0]}')
            self.main_logger.info(f'Using optimizers {self.ens_optimizers[0].__class__.__name__}')

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

        for name, metric in self.val_metrics.items():
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
            # checkpoint_save = {'model': self.model, 'optimizer': self.optimizer}
            checkpoint_save = {'model': self.model}

        best_ckpoint_handler = handlers.ModelCheckpoint(save_dir, 'best', n_saved=1, require_empty=False,
                                                        score_function=self.eval_func, save_as_state_dict=True)

        final_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpoint_handler, checkpoint_save)
        self.trainer.add_event_handler(Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def _init_early_stopping_handler(self) -> None:
        if self.train_cfg.early_stop:
            early_stop_handler = handlers.EarlyStopping(self.train_cfg.patience,
                                                        score_function=self.eval_func,
                                                        trainer=self.trainer)
            self.evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    def _on_epoch_started(self, _engine: Engine) -> None:
        if self.optim_cfg.scheduler == 'step':
            measure = _engine.state.epoch
        elif self.optim_cfg.scheduler == 'plateau':
            measure = 50 if _engine.state.epoch == 1 else self.val_loss(self.evaluator)
        else:
            return None

        def _update_individual_scheduler(lr_scheduler):
            lr_scheduler.step(measure)

        if self.use_ensemble:
            for lr_s in self.ens_lr_schedulers:
                _update_individual_scheduler(lr_s)
        else:
            _update_individual_scheduler(self.lr_scheduler)

    def _on_epoch_completed(self, _engine: Engine) -> None:
        self._log_training_results(_engine, self.main_logger, self.main_writer)
        self._evaluate_on_val(_engine, self.main_logger, self.main_writer)

    def _on_events_completed(self, _engine: Engine) -> None:
        self._finalize()

    def _on_exception_raised(self, _engine: Engine, e: Exception) -> None:
        self.main_logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.main_logger.info(e)
        self._finalize(True)
        raise e

    def _finalize(self, on_error=False) -> None:
        if self.trainer.should_terminate:
            self.main_logger.info(f'Early stopping on epoch {self.trainer.state.epoch}')

        self.main_writer.close()
        self.main_logger.removeHandler(self.main_log_handler)

    def _log_training_results(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        train_duration = timer_to_str(self.timer.value())
        avg_loss = _train_engine.state.metrics['train_loss']

        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}. ' \
              f'Duration: {train_duration} || Avg loss: {avg_loss:.4f}'
        logger.info(msg)

        writer.add_scalar('training/segment_loss', avg_loss, _train_engine.state.epoch)

    def _evaluate_on_val(self, _train_engine: Engine, logger: Logger, writer: SummaryWriter) -> None:
        self.evaluator.run(self.data_loaders.val_loader)

        eval_loss = self.evaluator.state.metrics['loss']
        segment_metrics = self.evaluator.state.metrics['segment_metrics']

        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}         ||         ' \
              f'IoU: {segment_metrics["avg_iou"]:.4f} | F1: {segment_metrics["avg_f1"]:.4f} | ' \
              f'mAP: {segment_metrics["mAP"]:.4f}'
        logger.info(msg)

        writer.add_scalar('validation/segment_loss', eval_loss, _train_engine.state.epoch)
        for key, value in segment_metrics.items():
            writer.add_scalar(f'validation_segment_metrics/{key}', value, _train_engine.state.epoch)

    @staticmethod
    def val_loss(_engine: Engine) -> float:
        return round(_engine.state.metrics['loss'], 6)

    @staticmethod
    def iou_score(_engine: Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_iou'], 6)

    @staticmethod
    def f1_score(_engine: Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_f1'], 6)

    def _init_handlers(self, _init_checkpoint=True) -> None:
        if _init_checkpoint:
            self._init_checkpoint_handler()

        self._init_epoch_timer()
        self._init_early_stopping_handler()

        self.trainer.add_event_handler(Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(Events.COMPLETED, self._on_events_completed)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception_raised)

    def run(self) -> None:
        self.main_logger.info(f'{self.log_name} initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)
