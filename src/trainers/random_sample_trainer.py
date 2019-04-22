import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer import ActiveTrainer


class RandomSampleTrainer(ActiveTrainer):
    """
    Implementation of AL trainer with Random Sampling acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(RandomSampleTrainer, self).__init__(config, save_dir, 'RandomSamplerTrainer')

    def _acquisition_function(self):
        new_files = np.random.choice(self.data_pool.data_pool, size=self.al_config.budget, replace=False).tolist()

        self._update_data_pool(new_files)
