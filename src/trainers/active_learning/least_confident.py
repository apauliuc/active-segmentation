import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer import ActiveTrainerScan


class LeastConfidentScan(ActiveTrainerScan):
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
        super(LeastConfidentScan, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        pred_dict = self._predict_proba(self.m_type)
        # pred_dict is dictionary of scan_id -> prediction as 3d tensor

        uncertainties = []
        for proba in pred_dict.values():
            unc = -(proba - 0.5).abs().mean()

            uncertainties.append(unc)

        data = np.array([np.arange(len(uncertainties))])
        data = np.append(data, np.array([uncertainties]), axis=0)

        sorted_uncertainty = data[:, data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
