from ignite import engine, handlers

from data import get_dataloaders
from helpers.config import ConfigClass
from helpers.utils import timer_to_str
from trainers.base_trainer import BaseTrainer


class PassiveTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(PassiveTrainer, self).__init__(config, save_dir, 'SimpleTrainer')
        self.epochs = self.train_cfg.num_epochs

        self.data_loaders = get_dataloaders(config.data)
        self.logger.info(self.data_loaders.msg)

        self._init_train_components()

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        self._log_training_results(_engine)
        self._run_evaluation(_engine)

    def _log_training_results(self, _train_engine: engine.Engine) -> None:
        train_duration = timer_to_str(self.timer.value())
        avg_loss = _train_engine.state.metrics['train_loss']
        msg = f'Training results - Epoch:{_train_engine.state.epoch:2d}/{_train_engine.state.max_epochs}. ' \
            f'Duration: {train_duration}. Avg loss: {avg_loss:.4f}'
        self.logger.info(msg)
        self.train_writer.add_scalar('training/avg_loss', avg_loss, _train_engine.state.epoch)

    def _run_evaluation(self, _train_engine: engine.Engine) -> None:
        self.evaluator.run(self.data_loaders.val_loader)
        eval_loss = self.evaluator.state.metrics['loss']
        eval_metrics = self.evaluator.state.metrics['segment_metrics']
        msg = f'Eval. on val_loader - Avg loss: {eval_loss:.4f}'
        self.logger.info(msg)
        self.train_writer.add_scalar('validation_eval/avg_loss', eval_loss, _train_engine.state.epoch)

        for key, value in eval_metrics.items():
            self.train_writer.add_scalar(f'val_metrics/{key}', value, _train_engine.state.epoch)

    def _init_handlers(self) -> None:
        self._init_epoch_timer()
        self._init_checkpoint_handler()

        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_events_completed)
        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, handlers.TerminateOnNan())

    def run(self) -> None:
        self.logger.info(f'SimpleTrainer initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.epochs)
