from data import get_dataloaders
from helpers.config import ConfigClass
from trainers.base_trainer import BaseTrainer


class PassiveTrainerEnsemble(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(PassiveTrainerEnsemble, self).__init__(config, save_dir, 'Passive_Trainer_Ensemble')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components_ensemble()
