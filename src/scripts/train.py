import os
import random
from typing import Tuple

import torch
import yaml
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from ignite import engine, handlers
from ignite import metrics

from data import get_dataloaders
from models import get_model
from losses import get_loss_function
from scripts.predict import main_predict
from optimizers import get_optimizer
from helpers.config import ConfigClass, get_config_from_path
from helpers.metrics import SegmentationMetrics
from helpers.utils import timer_to_str, setup_logger, retrieve_class_init_parameters
from helpers.paths import get_resume_model_path, get_resume_optimizer_path, get_new_run_path


# noinspection PyUnresolvedReferences
class Trainer:

    def __init__(self, config: ConfigClass, save_dir: str):
        self.save_dir = save_dir
        self.logger, self.log_handler = setup_logger(save_dir, 'scripts.train')
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

        self.data_loaders = get_dataloaders(data_cfg)
        self.logger.info(self.data_loaders.msg)

        self.metrics = {
            'loss': metrics.Loss(get_loss_function(train_cfg.loss_fn)),
            'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                   threshold=config.binarize_threshold)
        }

        model_cfg.network_params.input_channels = self.data_loaders.input_channels
        model_cfg.network_params.num_classes = self.data_loaders.num_classes

        self.model = self.__init__model(model_cfg)
        self.optimizer = self.__init_optimizer(optim_cfg)
        self.criterion = self.__init_criterion(train_cfg)

        self.lr_scheduler = self.__init_lr_scheduler(optim_cfg)

        self.trainer, self.evaluator = self.__init_engines()

        self.__init_handlers()

    def __init__model(self, model_cfg: ConfigClass):
        model = get_model(model_cfg).to(device=self.device)
        self.logger.info(f'Using model {model}')
        if self.resume_cfg.resume_from is not None:
            model_path = get_resume_model_path(self.resume_cfg.resume_from, self.resume_cfg.saved_model)
            self.logger.info(f'Loading model loaded from {model_path}')
            model.load_state_dict(torch.load(model_path))
        return model

    def __init_optimizer(self, optim_cfg: ConfigClass):
        optimizer_cls = get_optimizer(optim_cfg)

        init_param_names = retrieve_class_init_parameters(optimizer_cls)
        optimizer_params = {k: v for k, v in optim_cfg.items() if k in init_param_names}

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info(f'Using optimizer {optimizer.__class__.__name__}')

        if self.resume_cfg.resume_from is not None:
            optimizer_path = get_resume_optimizer_path(self.resume_cfg.resume_from, self.resume_cfg.saved_optimizer)
            self.logger.info(f'Loading optimizer from {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))
        return optimizer

    def __init_criterion(self, train_cfg: ConfigClass):
        criterion = get_loss_function(train_cfg.loss_fn).to(device=self.device)
        self.logger.info(f'Using loss function {criterion}')
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
        self.evaluator.run(self.data_loaders.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']
        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}'
        self.logger.info(msg)
        self.writer.add_scalar('validation_eval/avg_loss', eval_loss, _train_engine.state.epoch)

        for key, value in eval_metrics.items():
            self.writer.add_scalar(f'val_metrics/{key}', value, _train_engine.state.epoch)

    def _on_events_completed(self, _engine: engine.Engine) -> None:
        self.finalize()

    def _on_exception_raised(self, _engine: engine.Engine, e: Exception) -> None:
        self.logger.info(f'Exception at epoch {_engine.state.epoch}')
        self.logger.info(e)
        self.finalize()
        raise e

    # noinspection PyMethodMayBeStatic
    def val_loss(self, _engine: engine.Engine) -> float:
        return -round(_engine.state.metrics['loss'], 6)

    def finalize(self) -> None:
        self.writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.writer.close()
        self.logger.removeHandler(self.log_handler)

    def run(self) -> None:
        self.logger.info(f'All set. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.epochs)


def main_train_new_model(args, config_path: str):
    """Entry point for training a new model"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    trainer = Trainer(config, run_dir)
    trainer.run()

    if args.train_predict:
        main_predict(config, run_dir)
