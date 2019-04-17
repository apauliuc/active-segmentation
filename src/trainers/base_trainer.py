import abc
import os
import random
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
from helpers.utils import setup_logger, retrieve_class_init_parameters
from helpers.paths import get_resume_model_path, get_resume_optimizer_path


class BaseTrainer(abc.ABC):
    data_loaders: BaseLoader

    def __init__(self, config: ConfigClass, save_dir: str, log_name=''):
        self.save_dir = save_dir
        self.logger, self.log_handler = setup_logger(save_dir, log_name)
        self.logger.info(f'Saving to folder {save_dir}')
        self.train_writer = SummaryWriter(log_dir=save_dir)

        self.config = config
        self.model_cfg = config.model
        self.train_cfg = config.training
        self.optim_cfg = config.training.optimizer
        self.resume_cfg = config.resume

        if self.train_cfg.seed is not None:
            torch.manual_seed(self.train_cfg.seed)
            random.seed(self.train_cfg.seed)
            np.random.seed(self.train_cfg.seed)
            self.logger.info(f'Seed set on {self.train_cfg.seed}')

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
        self.logger.info(f'Using model {model}')

        if self.resume_cfg.resume_from is not None:
            model_path = get_resume_model_path(self.resume_cfg.resume_from, self.resume_cfg.saved_model)
            self.logger.info(f'Loading model loaded from {model_path}')
            model.load_state_dict(torch.load(model_path))

        return model

    def _init_optimizer(self):
        optimizer_cls = get_optimizer(self.optim_cfg)

        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in self.optim_cfg.items() if k in init_param_names}

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info(f'Using optimizer {optimizer.__class__.__name__}')

        if self.resume_cfg.resume_from is not None:
            optimizer_path = get_resume_optimizer_path(self.resume_cfg.resume_from, self.resume_cfg.saved_optimizer)
            self.logger.info(f'Loading optimizer from {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))

        return optimizer

    def _init_criterion(self):
        criterion = get_loss_function(self.train_cfg.loss_fn).to(device=self.device)
        self.logger.info(f'Using loss function {criterion}')

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

        best_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'best', n_saved=1, require_empty=False,
                                                           score_function=self.val_loss, save_as_state_dict=True)

        final_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, best_checkpoint_handler, checkpoint_save)
        self.trainer.add_event_handler(engine.Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(_engine.state.epoch)

    def _on_events_completed(self, _engine: engine.Engine) -> None:
        self._finalize()

    def _on_exception_raised(self, _engine: engine.Engine, e: Exception) -> None:
        self.logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.logger.info(e)
        self._finalize()
        raise e

    def _finalize(self) -> None:
        self.train_writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.train_writer.close()
        self.logger.removeHandler(self.log_handler)

    @staticmethod
    def val_loss(_engine: engine.Engine) -> float:
        return -round(_engine.state.metrics['loss'], 6)

    @abc.abstractmethod
    def _init_handlers(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass
