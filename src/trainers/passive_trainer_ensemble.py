from data import get_dataloaders
from helpers.config import ConfigClass
from trainers.base_trainer import BaseTrainer


class PassiveTrainerEnsemble(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(PassiveTrainerEnsemble, self).__init__(config, save_dir, 'Passive_Trainer_Ensemble')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components_ensemble()

    def run(self) -> None:
        self.main_logger.info(f'{self.log_name} initialised. Starting training on {self.device}.')
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)
