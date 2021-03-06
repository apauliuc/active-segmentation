import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer_img import ActiveTrainer


class LeastConfidentImage(ActiveTrainer):
    """
    Implementation of AL trainer with Least Confident acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'LeastConf_Ensemble_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'LeastConf_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(LeastConfidentImage, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        x = self._predict_proba(self.m_type)

        lc_values = -(x - 0.5).abs().mean(dim=1)

        unc_data = np.zeros((2, lc_values.shape[0]))
        unc_data[0] = np.arange(lc_values.shape[0])
        unc_data[1] = lc_values.numpy()

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_files_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_files = np.array(self.data_pool.unlabelled_files)[new_files_idx].tolist()

        self._update_data_pool(new_files)
