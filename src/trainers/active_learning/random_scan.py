import numpy as np

from helpers.config import ConfigClass
from trainers.active_learning.active_trainer_scan import ActiveTrainerScan


class RandomScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with Random Sampling acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(RandomScan, self).__init__(config, save_dir, 'Random_Scan_Trainer')

    def _acquisition_function(self):
        new_scans = np.random.choice(self.data_pool.unlabelled_scans,
                                     size=self.al_config.budget,
                                     replace=False).tolist()

        self._update_data_pool(new_scans)
