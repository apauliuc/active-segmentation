import os
import random
from typing import Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from ignite import engine, handlers
from ignite import metrics

from alsegment.data.medical_dataset import create_data_loader
from alsegment.helpers.config import ConfigClass
from alsegment.losses import get_loss_fn
from alsegment.models import get_model
from alsegment.helpers.metrics import SegmentationMetrics
from alsegment.helpers.utils import timer_to_str, setup_logger
from alsegment.helpers.paths import get_resume_model_path, get_resume_optimizer_path, get_dataset_path


class Trainer(object):

    def __init__(self, config: ConfigClass, save_dir: str):
        self.save_dir = save_dir
        self.logger, self.log_handler = setup_logger(save_dir)
        self.logger.info(f'Saving to folder {save_dir}')
        self.writer = SummaryWriter(log_dir=save_dir)

        data_cfg = config.data
        model_cfg = config.model
        train_cfg = config.training
        optim_cfg = train_cfg.optimizer
        self.resume_cfg = config.resume

        if train_cfg.seed is not None:
            torch.manual_seed(train_cfg.seed)
            random.seed(train_cfg.seed)
            self.logger.info(f'Seed set on {train_cfg.seed}')

        self.epochs = train_cfg.num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_train_loader = data_cfg.run_val_on_train

        self.metrics = {
            'loss': metrics.Loss(get_loss_fn(train_cfg.loss_fn)),
            'segment_metrics': SegmentationMetrics()
        }

        self.train_loader, self.val_loader, self.val_train_loader = self.__init_data_loaders(data_cfg)

        self.model = self.__init__model(model_cfg)
        self.optimizer = self.__init_optimizer(optim_cfg)
        self.criterion = self.__init_criterion(train_cfg)

        self.lr_scheduler = self.__init_lr_scheduler(optim_cfg)

        self.trainer, self.evaluator = self.__init_engines()

        self.__init_handlers()

    def __init_data_loaders(self, data_cfg: ConfigClass):
        train_path = get_dataset_path(data_cfg.path, data_cfg.dataset, data_cfg.train_split)
        train_loader = create_data_loader(data_cfg, train_path, batch_size=data_cfg.batch_size)
        self.logger.info(f'Train data loader created from {train_path}')

        val_path = get_dataset_path(data_cfg.path, data_cfg.dataset, data_cfg.val_split)
        val_loader = create_data_loader(data_cfg, val_path, batch_size=data_cfg.batch_size_val)
        self.logger.info(f'Validation data loader created from {val_path}')

        if self.eval_train_loader:
            val_train_loader = create_data_loader(data_cfg, train_path, batch_size=data_cfg.batch_size_val)
        else:
            val_train_loader = None

        return train_loader, val_loader, val_train_loader

    def __init__model(self, model_cfg: ConfigClass):
        model = get_model(model_cfg).to(device=self.device)
        if self.resume_cfg.resume_from is not None:
            model_path = get_resume_model_path(self.resume_cfg.resume_from, self.resume_cfg.saved_model)
            self.logger.info(f'Loading model loaded from {model_path}')
            model.load_state_dict(torch.load(model_path))
        return model

    def __init_optimizer(self, optim_cfg: ConfigClass):
        optimizer = optim.Adam(self.model.parameters(), lr=optim_cfg.lr,
                               weight_decay=optim_cfg.weight_decay, amsgrad=optim_cfg.amsgrad)

        if self.resume_cfg.resume_from is not None:
            optimizer_path = get_resume_optimizer_path(self.resume_cfg.resume_from, self.resume_cfg.saved_optimizer)
            self.logger.info(f'Loading optimizer from {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))
        return optimizer

    def __init_criterion(self, train_cfg: ConfigClass):
        criterion = get_loss_fn(train_cfg.loss_fn).to(device=self.device)
        return criterion

    def __init_lr_scheduler(self, optim_cfg: ConfigClass):
        lr_scheduler = None
        if optim_cfg.scheduler == 'step':
            lr_scheduler = StepLR(self.optimizer, step_size=optim_cfg.lr_cycle, gamma=0.1)
        return lr_scheduler

    def __init_engines(self) -> Tuple[engine.Engine, engine.Engine]:
        trainer = engine.create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)
        evaluator = engine.create_supervised_evaluator(self.model, self.metrics, self.device, True)

        metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')

        return trainer, evaluator

    def __init_handlers(self) -> None:
        self.__init_iter_timer()
        self.__init_checkpoint_handler()

        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_events_completed)
        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

    def __init_checkpoint_handler(self) -> None:
        checkpoint_save = {
            'model': self.model,
            'optimizer': self.optimizer
        }

        best_checkpoint_handler = handlers.ModelCheckpoint(self.save_dir, 'best', n_saved=1, require_empty=False,
                                                           score_function=self.val_loss, save_as_state_dict=True)

        final_checkpoint_handler = handlers.ModelCheckpoint(self.save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, best_checkpoint_handler, checkpoint_save)
        self.trainer.add_event_handler(engine.Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def __init_iter_timer(self) -> None:
        self.timer = handlers.Timer(average=False)
        self.timer.attach(self.trainer,
                          start=engine.Events.EPOCH_STARTED,
                          resume=engine.Events.ITERATION_STARTED,
                          pause=engine.Events.ITERATION_COMPLETED)

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        self._log_training_results(_engine)
        self._run_evaluation(_engine)

    def _log_training_results(self, _train_engine: engine.Engine) -> None:
        train_duration = timer_to_str(self.timer.value())
        avg_loss = _train_engine.state.metrics['train_loss']
        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}. ' \
            f'Duration: {train_duration}. Avg loss: {avg_loss:.4f}'
        self.logger.info(msg)
        self.writer.add_scalar('training/avg_loss', avg_loss, _train_engine.state.epoch)
        self.writer.add_scalar('training/train_timer', self.timer.value(), _train_engine.state.epoch)

    def _run_evaluation(self, _train_engine: engine.Engine) -> None:
        self.evaluator.run(self.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']
        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}'
        self.logger.info(msg)
        self.writer.add_scalar('validation_eval/avg_loss', eval_loss, _train_engine.state.epoch)

        for key, value in eval_metrics.items():
            self.writer.add_scalar(f'val_metrics/{key}', value, _train_engine.state.epoch)

    def _on_events_completed(self, _engine: engine.Engine) -> None:
        self.writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.writer.close()

    def _on_exception_raised(self, _engine: engine.Engine, e: Exception) -> None:
        self.logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.logger.info(e)
        self._on_events_completed(_engine)
        raise e

    # noinspection PyMethodMayBeStatic
    def val_loss(self, _engine: engine.Engine) -> float:
        return -round(_engine.state.metrics['loss'], 6)

    def run(self) -> None:
        self.logger.info(f'All set. Starting training on {self.device}.')
        self.trainer.run(self.train_loader, max_epochs=self.epochs)

        self.logger.removeHandler(self.log_handler)
