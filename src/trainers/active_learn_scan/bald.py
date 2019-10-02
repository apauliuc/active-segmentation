import numpy as np
import torch

from helpers.config import ConfigClass
from trainers.active_trainer_scan import ActiveTrainerScan


class BALDScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with BALD acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'BALD_Ensemble_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'BALD_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(BALDScan, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        pred_dict = self._predict_proba_individual(self.m_type)
        # pred_dict is dictionary of scan_id -> list of predictions as 2d tensors

        bald_values = []
        for predictions_list in pred_dict.values():
            avg_prediction = torch.zeros_like(predictions_list[0])
            sum_individual_entropies = np.zeros_like(predictions_list[0])

            for prediction in predictions_list:
                avg_prediction += prediction

                individual_entropy = self._compute_pixel_entropy(prediction.numpy())
                sum_individual_entropies += individual_entropy

            # Compute average of MC prediction entropies
            sum_individual_entropies = sum_individual_entropies / len(predictions_list)

            # Compute entropy of MC averaged prediction
            avg_prediction = avg_prediction / len(predictions_list)
            entropy_avg_prediction = self._compute_pixel_entropy(avg_prediction.numpy())

            # Combine measures
            bald = sum_individual_entropies + entropy_avg_prediction

            bald_values.append(bald.mean())

        unc_data = np.array([np.arange(len(bald_values))])
        unc_data = np.append(unc_data, np.array([bald_values]), axis=0)

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_scans_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_scans = np.array(self.data_pool.unlabelled_scans)[new_scans_idx].tolist()

        self._update_data_pool(new_scans)
