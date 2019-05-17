import numpy as np

from helpers.config import ConfigClass
from trainers.active_learning.active_trainer import ActiveTrainerScan


class LeastConfidentScanMC(ActiveTrainerScan):
    """
    Implementation of MC dropout AL trainer with Least Confident acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(LeastConfidentScanMC, self).__init__(config, save_dir, 'LC_Scan_MC_Trainer')

    def _acquisition_function(self):
        pred_dict = self._predict_proba_mc_dropout()
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
