import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer_img import ActiveTrainer


class LeastConfident(ActiveTrainer):
    """
    Implementation of AL trainer with Least Confident acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'LC_Ensemble_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'LC_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(LeastConfident, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        x = self._predict_proba(self.m_type)

        x = -(x - 0.5).abs().mean(dim=1)

        data = np.zeros((2, x.shape[0]))
        data[0] = np.arange(x.shape[0])
        data[1] = x.numpy()

        sorted_uncertainty = data[:, data[1, :].argsort()[::-1]]

        new_files_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_files = np.array(self.data_pool.unlabelled_files)[new_files_idx].tolist()

        self._update_data_pool(new_files)
