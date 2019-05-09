import os
import torch

from ignite import handlers
from ignite import engine
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from helpers.torch_utils import apply_dropout
from trainers.base_trainer import BaseTrainer
from data import MDSDataLoaders
from alsegment.data_pool import ALDataPool
from helpers.config import ConfigClass
from helpers.utils import setup_logger


class ActiveTrainer(BaseTrainer):
    """
    Base implementation of AL trainer.
    """
    data_loaders: MDSDataLoaders
    acquisition_step: int

    def __init__(self, config: ConfigClass, save_dir: str, name='ActiveTrainer'):
        super(ActiveTrainer, self).__init__(config, save_dir, name)

        self.main_data_dir = os.path.join(self.save_dir, 'Datasets')
        os.makedirs(self.main_data_dir)

        self.al_config = config.active_learn
        self._create_train_loggers(value=0)

        self.data_pool = ALDataPool(config)
        self.data_loaders = MDSDataLoaders(self.config.data, file_list=self.data_pool.train_pool)
        self.main_logger.info(self.data_loaders.msg)

        self.data_pool.copy_pool_files_to_dir(self.data_pool.train_pool, self.save_data_dir)

        self._init_train_components()

    def _create_train_loggers(self, value):
        self.acquisition_step = value
        self.save_model_dir = os.path.join(self.save_dir, f'Step {value}')
        os.makedirs(self.save_model_dir)
        self.save_data_dir = os.path.join(self.main_data_dir, f'Step {value}')
        os.makedirs(self.save_data_dir)

        self.train_logger, self.train_log_handler = setup_logger(self.save_model_dir, f'Train step {value}')
        self.train_writer = SummaryWriter(log_dir=self.save_model_dir)

    def _update_components_on_step(self, value):
        self._create_train_loggers(value=value)

        # Update optimizer learning rate and recreate LR scheduler
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.optim_cfg.lr
        self.lr_scheduler = self._init_lr_scheduler()

        # Recreate Engines and handlers
        self.trainer, self.evaluator = self._init_engines()
        self._init_handlers()

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        self._log_training_results(_engine, self.train_logger, self.train_writer)
        self._evaluate_on_val(_engine, self.train_logger, self.train_writer)

    def _init_handlers(self) -> None:
        self._init_epoch_timer()
        self._init_checkpoint_handler(save_dir=self.save_model_dir)
        self._init_early_stopping_handler()

        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_events_completed)

        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

    def _finalize(self) -> None:
        if self.trainer.should_terminate:
            self.train_logger.info(f'Early stopping on epoch {self.trainer.state.epoch}')

        # Evaluate model and save information
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']

        msg = f'Step {self.acquisition_step} - Avg. validation loss after ' \
            f'{self.trainer.state.epoch} training epochs: {eval_loss:.4f}'
        self.main_logger.info(msg)

        self.main_writer.add_scalar(f'active_learning/avg_val_loss', eval_loss, self.acquisition_step)
        for key, value in eval_metrics.items():
            self.main_writer.add_scalar(f'active_learning/{key}', value, self.acquisition_step)
        self.main_writer.add_scalar(f'active_learning/epochs_trained', self.trainer.state.epoch, self.acquisition_step)

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
        self.main_logger.info(f'Training - acquisition step 0')
        self.main_logger.info(f'Using {len(self.data_pool.train_pool)} datapoints.')
        self._train()

        for i in range(1, self.al_config.acquisition_steps + 1):
            if len(self.data_pool.data_pool) < self.al_config.budget:
                self.main_logger.info(f'Data pool too small. Stopping training.')
                break

            self.main_logger.info(f'Training - acquisition step {i}')
            self._update_components_on_step(i)

            self._acquisition_function()

            self.main_logger.info(f'Using {len(self.data_pool.train_pool)} datapoints.')
            self._train()

        self._finalize_trainer()

    def _update_data_pool(self, new_data_points: list):
        self.data_pool.update_train_pool(new_data_points)
        self.data_loaders.update_train_loader(self.data_pool.train_pool)

        self.data_pool.copy_pool_files_to_dir(new_data_points, self.save_data_dir)

    def _acquisition_function(self) -> None:
        pass

    def _predict_proba(self):
        al_loader = DataLoader(self.data_pool,
                               batch_size=self.config.data.batch_size_val,
                               shuffle=False,
                               num_workers=self.config.data.num_workers,
                               pin_memory=torch.cuda.is_available())
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in al_loader:
                x, _ = batch
                x = x.to(self.device)

                out = self.model(x)
                out_probas = torch.sigmoid(out).reshape((out.shape[0], -1))

                predictions.extend(*out_probas.split(out_probas.shape[0]))

        return torch.stack(predictions)

    def _predict_proba_mc_dropout(self):
        al_loader = DataLoader(self.data_pool,
                               batch_size=self.config.data.batch_size_val,
                               shuffle=False,
                               num_workers=self.config.data.num_workers,
                               pin_memory=torch.cuda.is_available())

        mc_probas = torch.zeros((len(al_loader.dataset), 262144)).to(device=self.device)

        self.model.eval()
        self.model.apply(apply_dropout)
        for i in range(self.al_config.mc_passes):
            with torch.no_grad():
                for batch in al_loader:
                    x, idxs = batch
                    x = x.to(self.device)

                    out = self.model(x)
                    out_probas = torch.sigmoid(out).reshape((out.shape[0], -1))

                    mc_probas[idxs] += out_probas

        mc_probas = mc_probas / self.al_config.mc_passes
        return mc_probas
