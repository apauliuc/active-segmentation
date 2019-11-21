import pickle
from os.path import join

from data import get_dataloaders
from definitions import DATA_DIR
from helpers.config import ConfigClass
from trainers.base_trainer import BaseTrainer


class PassiveTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(PassiveTrainer, self).__init__(config, save_dir, 'Passive_Trainer')

        if config.data.data_list is None:
            self.data_loaders = get_dataloaders(config.data)
        else:
            with open(join(DATA_DIR, 'MSRA10K_INIT', config.data.data_list), 'rb') as f:
                file_list = pickle.load(f)
            self.data_loaders = get_dataloaders(config.data, file_list=file_list)
            self.main_logger.info(f'Using preset file list with length {len(file_list)}')
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components()
