import numpy as np
import torch

from helpers.config import ConfigClass
from trainers.active_trainer_img import ActiveTrainer


class BALDCombinedImage(ActiveTrainer):
    """
    Implementation of AL trainer with BALD acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        name = 'BALD_Combined_Trainer'
        super(BALDCombinedImage, self).__init__(config, save_dir, name)

    def _acquisition_function(self):
        predictions_list = self._predict_proba_combined_method()
        # predictions_list is list of 2D tensors

        avg_prediction = torch.zeros_like(predictions_list[0])
        sum_individual_entropies = np.zeros_like(predictions_list[0])

        for prediction in predictions_list:
            avg_prediction += prediction

            individual_ent = self._compute_pixel_entropy(prediction.numpy())
            sum_individual_entropies += individual_ent

        # Average of prediction entropies
        sum_individual_entropies = sum_individual_entropies / len(predictions_list)

        # Compute entropy of MC averaged prediction
        avg_prediction = avg_prediction / len(predictions_list)
        entropy_avg_prediction = self._compute_pixel_entropy(avg_prediction.numpy())

        # Combine measures
        bald = sum_individual_entropies + entropy_avg_prediction
        bald = bald.mean(axis=1)

        unc_data = np.zeros((2, bald.shape[0]))
        unc_data[0] = np.arange(bald.shape[0])
        unc_data[1] = bald

        sorted_uncertainty = unc_data[:, unc_data[1, :].argsort()[::-1]]

        new_files_idx = sorted_uncertainty[0, :self.al_config.budget].astype(np.int)
        new_files = np.array(self.data_pool.unlabelled_files)[new_files_idx].tolist()

        self._update_data_pool(new_files)
