from ignite.engine.engine import Engine
from ignite.engine import create_supervised_trainer, create_supervised_evaluator

from data import get_dataloaders
from helpers.config import ConfigClass
from trainers.base_trainer import BaseTrainer


class VariationalTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(VariationalTrainer, self).__init__(config, save_dir, 'Variational_Trainer')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components()

    def _init_trainer_engine(self) -> Engine:
        return create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)

    def _init_evaluator_engine(self) -> Engine:
        return create_supervised_evaluator(self.model, self.metrics, self.device, True)

    def run(self) -> None:
        self.main_logger.info(f'{self.log_name} initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)
