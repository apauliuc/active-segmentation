import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer import ActiveTrainerScan


class WeightedLeastConfidentScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with Least Confident acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'Weighted_LC_Ensemble_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'Weighted_LC_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(WeightedLeastConfidentScan, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        pred_dict, weights = self._predict_proba(self.m_type, retrieve_weights=True)
        # pred_dict is dictionary of scan_id -> prediction as 2d numpy array

        uncertainties = []
        for key, proba in pred_dict.items():
            img_unc = np.abs(-(proba - 0.5)).mean(axis=1)

            unc = (img_unc * weights[key]).sum()

            uncertainties.append(unc)

        data = np.array([np.arange(len(uncertainties))])
        data = np.append(data, np.array([uncertainties]), axis=0)

        sorted_uncertainty = data[:, data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
