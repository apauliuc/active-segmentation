import torch
from ignite.engine.engine import Engine

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
        self.model.to(self.device)

        # noinspection PyUnusedLocal
        def _update(_engine, batch):
            self.model.train()
            self.optimizer.zero_grad()

            x, y = batch
            x = x.to(device=self.device, non_blocking=True)
            y = y.to(device=self.device, non_blocking=True)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()

            self.optimizer.step()
            return loss.item()

        return Engine(_update)

    def _init_evaluator_engine(self) -> Engine:
        self.model.to(self.device)

        # noinspection PyUnusedLocal
        def _inference(_engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch
                x = x.to(device=self.device, non_blocking=True)
                y = y.to(device=self.device, non_blocking=True)

                y_pred = self.model(x)
                return y_pred, y

        engine = Engine(_inference)

        for name, metric in self.metrics.items():
            metric.attach(engine, name)

        return engine

    def run(self) -> None:
        self.main_logger.info(f'{self.log_name} initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)
