from ignite import handlers
from ignite.engine.engine import Events

from data import get_dataloaders
from helpers.config import ConfigClass
from trainers.base_trainer import BaseTrainer


class PassiveTrainerEnsemble(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(PassiveTrainerEnsemble, self).__init__(config, save_dir, 'Passive_Trainer_Ensemble')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components_ensemble()

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

    def run(self) -> None:
        self.main_logger.info(f'Passive_Trainer_Ensemble initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)
