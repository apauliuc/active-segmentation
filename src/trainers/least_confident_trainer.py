import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer import ActiveTrainer


class LeastConfidentTrainer(ActiveTrainer):
    """
    Implementation of AL trainer with Least Confident acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        name = 'LeastConfidentMCTrainer' if 'mc' in self.config.active_learn.method else 'LeastConfidentTrainer'
        super(LeastConfidentTrainer, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        if 'mc' in self.config.active_learn.method:
            x = self._predict_proba_mc_dropout().cpu()
        else:
            x = self._predict_proba().cpu()

        x = -(x - 0.5).abs().mean(dim=1)

        data = np.zeros((2, x.shape[0]))
        data[0] = np.arange(x.shape[0])
        data[1] = x.numpy()

        sorted_uncertainty = data[:, data[1, :].argsort()[::-1]]

        new_files_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_files = np.array(self.data_pool.data_pool)[new_files_idx].tolist()

        self._update_data_pool(new_files)
