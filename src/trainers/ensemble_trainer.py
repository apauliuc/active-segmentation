import os

import torch
from ignite.engine.engine import Engine, Events
from ignite import handlers
from ignite import metrics
from torch.optim.lr_scheduler import StepLR

from data import get_dataloaders
from helpers.config import ConfigClass
from helpers.metrics import SegmentationMetrics
from helpers.utils import retrieve_class_init_parameters
from losses.bce_and_jaccard_ensemble import BCEAndJaccardLossEnsemble
from models import get_model
from optimizers import get_optimizer
from trainers.base_trainer import BaseTrainer


class EnsembleTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(EnsembleTrainer, self).__init__(config, save_dir, 'EnsembleTrainer')
        self.ensemble_cfg = self.train_cfg.ensemble

        self.epochs = self.train_cfg.num_epochs
        self.len_models = self.ensemble_cfg.number_models

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components()

    def _init_train_components(self):
        self.metrics = {
            'loss': metrics.Loss(BCEAndJaccardLossEnsemble()),
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

        self.trainer = self._init_trainer_engine()
        self.evaluator = self._init_evaluator_engine()

        metrics.RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'train_loss')

        self._init_handlers()

    def _init_trainer_engine(self) -> Engine:
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

    def _init_evaluator_engine(self) -> Engine:
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

    def _init_checkpoint_handler(self, save_dir=None) -> None:
        save_dir = self.save_dir if save_dir is None else save_dir

        checkpoint_save = {f'model_{i}': model for i, model in enumerate(self.ens_models)}

        best_ckpoint = handlers.ModelCheckpoint(save_dir, 'best', n_saved=1, require_empty=False,
                                                score_function=self.eval_func, save_as_state_dict=True)
        final_checkpoint_handler = handlers.ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1,
                                                            require_empty=False, save_as_state_dict=True)

        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpoint, checkpoint_save)
        self.trainer.add_event_handler(Events.COMPLETED, final_checkpoint_handler, checkpoint_save)

    def _on_epoch_started(self, _engine: Engine) -> None:
        for lr_scheduler in self.ens_lr_schedulers:
            if lr_scheduler is not None:
                lr_scheduler.step(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: Engine) -> None:
        self._log_training_results(_engine, self.main_logger, self.main_writer)
        self._evaluate_on_val(_engine, self.main_logger, self.main_writer)

    def _init_handlers(self) -> None:
        self._init_epoch_timer()
        self._init_checkpoint_handler()
        self._init_early_stopping_handler()

        self.trainer.add_event_handler(Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(Events.COMPLETED, self._on_events_completed)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

    def _finalize(self) -> None:
        if self.trainer.should_terminate:
            self.main_logger.info(f'Early stopping on epoch {self.trainer.state.epoch}')

        self.main_writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.main_writer.close()
        self.main_logger.removeHandler(self.main_log_handler)

    def run(self) -> None:
        self.main_logger.info(f'EnsembleTrainer initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.epochs)
