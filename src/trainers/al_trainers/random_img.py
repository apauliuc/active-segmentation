import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer_img import ActiveTrainer


class Random(ActiveTrainer):
    """
    Implementation of AL trainer with Random Sampling acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(Random, self).__init__(config, save_dir, 'Random_Trainer')

    def _acquisition_function(self):
        new_files = np.random.choice(self.data_pool.unlabelled_files,
                                     size=self.al_config.budget,
                                     replace=False).tolist()

        self._update_data_pool(new_files)
