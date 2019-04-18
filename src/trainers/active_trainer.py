import os
import numpy as np

from ignite import handlers
from ignite import engine
from tensorboardX import SummaryWriter

from trainers.base_trainer import BaseTrainer
from data import MDSDataLoaders
from alsegment.data_pool import ALDataPool
from helpers.config import ConfigClass
from helpers.utils import setup_logger


class ActiveTrainer(BaseTrainer):
    data_loaders: MDSDataLoaders
    acquisition_step: int

    def __init__(self, config: ConfigClass, save_dir: str):
        super(ActiveTrainer, self).__init__(config, save_dir, 'ActiveTrainer')

        self.al_config = config.active_learn
        self._create_train_loggers(value=0)

        self.data_pool = ALDataPool(config)
        self.data_loaders = MDSDataLoaders(self.config.data, file_list=self.data_pool.train_pool)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components()

    def _create_train_loggers(self, value):
        self.acquisition_step = value
        self.save_model_dir = os.path.join(self.save_dir, f'Step {value}')
        os.makedirs(self.save_model_dir)

        self.train_logger, self.train_log_handler = setup_logger(self.save_model_dir, f'Train step {value}')
        self.train_writer = SummaryWriter(log_dir=self.save_model_dir)

    def _update_components_on_step(self, value):
        self._create_train_loggers(value=value)

        # Recreate Engines and handlers
        # TODO: check whether model is reinitialised
        # TODO: check whether optimizer needs to be reinitialised (or lr set back to initial)
        self.trainer, self.evaluator = self._init_engines()
        self._init_handlers()

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        self._log_training_results(_engine, self.train_logger, self.train_writer)
        self._evaluate_on_val(_engine, self.train_logger, self.train_writer)

    def _init_handlers(self) -> None:
        self._init_epoch_timer()
        self._init_checkpoint_handler(save_dir=self.save_model_dir)

        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_events_completed)

        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

    def _finalize(self) -> None:
        # Evaluate model and save information
        # TODO: evaluator already has state information. Rerun on val dataset not necessary
        # self.evaluator.run(self.data_loaders.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']

        msg = f'Step {self.acquisition_step} - Avg. validation loss: {eval_loss:.4f}'
        self.main_logger.info(msg)
        self.main_writer.add_scalar(f'active_learning/avg_val_loss', eval_loss, self.acquisition_step)
        for key, value in eval_metrics.items():
            self.main_writer.add_scalar(f'active_learning/{key}', value, self.acquisition_step)

        # Close writer and logger related to model training
        self.train_writer.export_scalars_to_json(os.path.join(self.save_model_dir, 'tensorboardX.json'))
        self.train_writer.close()
        self.train_logger.removeHandler(self.train_log_handler)

    def _finalize_trainer(self) -> None:
        # Close writer and logger related to trainer class
        self.main_writer.export_scalars_to_json(os.path.join(self.save_dir, 'tensorboardX.json'))
        self.main_writer.close()
        self.main_logger.removeHandler(self.main_log_handler)

    def _train(self) -> None:
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)

    def run(self) -> None:
        self.main_logger.info(f'ActiveTrainer initialised. Starting training on {self.device}.')
        self.main_logger.info('Training - acquisition step 0')
        self._train()

        for i in range(1, self.al_config.acquisition_steps):
            self.main_logger.info(f'Training - acquisition step {i}')
            self._update_components_on_step(i)

            self._query_new_data()

            self._train()

        self._finalize_trainer()

    def _update_data(self, new_data_points: list):
        self.data_pool.update_train_pool(new_data_points)
        self.data_loaders.update_train_loader(self.data_pool.train_pool)

    def _query_new_data(self) -> None:
        # For now, take random
        new_files = np.random.choice(self.data_pool.data_pool, size=self.al_config.budget, replace=False).tolist()
        self._update_data(new_files)
