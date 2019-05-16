import numpy as np
from scipy.special import xlogy

from helpers.config import ConfigClass
from trainers.al_scan_trainers.active_trainer_scan import ActiveTrainerScan


class BALDScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with BALD acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        super(BALDScan, self).__init__(config, save_dir, 'BALD_Trainer')

    @staticmethod
    def _compute_entropy(x):
        proba = np.expand_dims(x, 0)
        p = np.concatenate([proba, 1 - proba])
        logp = xlogy(np.sign(p), p) / np.log(2)
        return -np.nansum(p * logp, axis=0)

    def _acquisition_function(self):
        pred_dict = self._predict_proba_mc_dropout_individual()
        # pred_dict is dictionary of scan_id -> list of predictions as 3d tensors

        bald_values = []
        for prediction_list in pred_dict.values():
            mc_prediction = None
            entropy_mc_predictions = None

            for prediction in prediction_list:
                if mc_prediction is None:
                    mc_prediction = prediction
                else:
                    mc_prediction += prediction

                individual_entropy = self._compute_entropy(prediction.numpy())

                if entropy_mc_predictions is None:
                    entropy_mc_predictions = individual_entropy
                else:
                    entropy_mc_predictions += individual_entropy

            # Compute average of MC prediction entropies
            entropy_mc_predictions = entropy_mc_predictions / len(prediction_list)

            # Compute entropy of MC averaged prediction
            mc_prediction = mc_prediction / len(prediction_list)
            mc_averaged_entropy = self._compute_entropy(mc_prediction.numpy())

            # Combine measures
            bald = entropy_mc_predictions + mc_averaged_entropy

            bald_values.append(bald.mean())

        unc_data = np.array([np.arange(len(bald_values))])
        unc_data = np.append(unc_data, np.array([bald_values]), axis=0)

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
