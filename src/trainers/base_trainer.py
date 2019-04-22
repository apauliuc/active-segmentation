import abc
import random
from logging import Logger

import numpy as np
from typing import Tuple

import torch
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from ignite import engine, handlers
from ignite import metrics

from data.base_loader import BaseLoader
from models import get_model
from losses import get_loss_function
from optimizers import get_optimizer
from helpers.config import ConfigClass
from helpers.metrics import SegmentationMetrics
from helpers.utils import setup_logger, retrieve_class_init_parameters, timer_to_str
from helpers.paths import get_resume_model_path, get_resume_optimizer_path


class BaseTrainer(abc.ABC):
    data_loaders: BaseLoader

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

    def _init_engines(self) -> Tuple[engine.Engine, engine.Engine]:
        trainer = engine.create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)
        evaluator = engine.create_supervised_evaluator(self.model, self.metrics, self.device, True)

        metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')

        return trainer, evaluator

    def _init_epoch_timer(self) -> None:
        self.timer = handlers.Timer(average=False)
        self.timer.attach(self.trainer,
                          start=engine.Events.EPOCH_STARTED,
                          resume=engine.Events.ITERATION_STARTED,
                          pause=engine.Events.ITERATION_COMPLETED)

    def _init_checkpoint_handler(self, save_dir=None) -> None:
        save_dir = self.save_dir if save_dir is None else save_dir

        checkpoint_save = {
            'model': self.model,
            'optimizer': self.optimizer
        }

        best_loss_ckpoint = handlers.ModelCheckpoint(save_dir, 'best_loss', n_saved=1, require_empty=False,
                                                     score_function=self.val_loss, save_as_state_dict=True)
        best_iou_ckpoint = handlers.ModelCheckpoint(save_dir, 'best_iou', n_saved=1, require_empty=False,
                                                    score_function=self.iou_score, save_as_state_dict=True)
        best_f1_ckpoint = handlers.ModelCheckpoint(save_dir, 'best_f1', n_saved=1, require_empty=False,
                                                   score_function=self.f1_score, save_as_state_dict=True)

        final_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, best_loss_ckpoint, {'model': self.model})
        self.evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, best_iou_ckpoint, {'model': self.model})
        self.evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, best_f1_ckpoint, {'model': self.model})
        self.trainer.add_event_handler(engine.Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def _init_early_stopping_handler(self) -> None:
        if self.train_cfg.early_stop_fn == 'f1_score':
            func = self.f1_score
        elif self.train_cfg.early_stop_fn == 'iou_score':
            func = self.iou_score
        else:
            func = self.val_loss

        early_stop_handler = handlers.EarlyStopping(self.train_cfg.patience,
                                                    score_function=func,
                                                    trainer=self.trainer)
        self.evaluator.add_event_handler(engine.Events.COMPLETED, early_stop_handler)

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        pass

    def _on_events_completed(self, _engine: engine.Engine) -> None:
        self._finalize()

    def _on_exception_raised(self, _engine: engine.Engine, e: Exception) -> None:
        self.main_logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.main_logger.info(e)
        self._finalize()
        raise e

    def _finalize(self) -> None:
        pass

    def _finalize_trainer(self) -> None:
        pass

    def _log_training_results(self, _train_engine: engine.Engine, logger: Logger, writer: SummaryWriter) -> None:
        train_duration = timer_to_str(self.timer.value())
        avg_loss = _train_engine.state.metrics['train_loss']
        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}. ' \
            f'Duration: {train_duration}. Avg loss: {avg_loss:.4f}'
        logger.info(msg)
        writer.add_scalar('training/avg_loss', avg_loss, _train_engine.state.epoch)

    def _evaluate_on_val(self, _train_engine: engine.Engine, logger: Logger, writer: SummaryWriter) -> None:
        self.evaluator.run(self.data_loaders.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']
        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}'
        logger.info(msg)
        writer.add_scalar('validation_eval/avg_loss', eval_loss, _train_engine.state.epoch)

        for key, value in eval_metrics.items():
            writer.add_scalar(f'val_metrics/{key}', value, _train_engine.state.epoch)

    @staticmethod
    def val_loss(_engine: engine.Engine) -> float:
        return -round(_engine.state.metrics['loss'], 6)

    @staticmethod
    def iou_score(_engine: engine.Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_iou'].item(), 6)

    @staticmethod
    def f1_score(_engine: engine.Engine) -> float:
        return round(_engine.state.metrics['segment_metrics']['avg_f1'].item(), 6)

    @abc.abstractmethod
    def _init_handlers(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass
