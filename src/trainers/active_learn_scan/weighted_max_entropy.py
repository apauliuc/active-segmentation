import numpy as np

from helpers.config import ConfigClass
from trainers.active_trainer_scan import ActiveTrainerScan


class WeightedMaxEntropyScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with MaxEntropy acquisition function according to Shannon's entropy
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'Weighted_MaxEntropy_Ensemble_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'Weighted_MaxEntropy_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(WeightedMaxEntropyScan, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        pred_dict, weights = self._predict_proba(self.m_type, retrieve_weights=True)
        # pred_dict is dictionary of scan_id -> prediction as 2d numpy array

        entropy_values = []
        for key, proba in pred_dict.items():
            entropy_per_image = self._compute_pixel_entropy(proba).mean(axis=1)

            unc = (entropy_per_image * weights[key]).sum()

            entropy_values.append(unc)

        unc_data = np.array([np.arange(len(entropy_values))])
        unc_data = np.append(unc_data, np.array([entropy_values]), axis=0)

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
