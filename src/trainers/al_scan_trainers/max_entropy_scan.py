import numpy as np
from scipy.special import xlogy

from helpers.config import ConfigClass
from trainers.al_scan_trainers.active_trainer_scan import ActiveTrainerScan


class MaxEntropyScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with MaxEntropy acquisition function according to Shannon's entropy
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(MaxEntropyScan, self).__init__(config, save_dir, 'MaxEntropy_Trainer')

    def _acquisition_function(self):
        pred_dict = self._predict_proba_mc_dropout()
        # pred_dict is dictionary of scan_id -> prediction as 3d tensor

        entropy_values = []
        for proba in pred_dict.values():
            proba = np.expand_dims(proba.numpy(), 0)

            p = np.concatenate([proba, 1-proba])
            logp = xlogy(np.sign(p), p) / np.log(2)

            entropies = -np.nansum(p * logp, axis=0)
            entropy_values.append(entropies.mean())

        unc_data = np.array([np.arange(len(entropy_values))])
        unc_data = np.append(unc_data, np.array([entropy_values]), axis=0)

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
