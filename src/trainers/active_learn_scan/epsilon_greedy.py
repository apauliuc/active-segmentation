import numpy as np
import torch

from helpers.config import ConfigClass
from trainers.active_trainer_scan import ActiveTrainerScan


class EpsilonGreedyTrainerScan(ActiveTrainerScan):
    """
    Implementation of AL trainer with Epsilon Greedy acquisition function
    """

    def __init__(self, config: ConfigClass, save_dir: str):
        if config.training.use_ensemble:
            name = 'Eps_Greedy_Ens_Trainer'
            self.m_type = 'ensemble'
        else:
            name = 'Eps_Greedy_MC_Trainer'
            self.m_type = 'mc_dropout'
        super(EpsilonGreedyTrainerScan, self).__init__(config, save_dir, name)

        self.epsilon = 0.5

    def _random_acquisition(self):
        new_scans = np.random.choice(self.data_pool.unlabelled_scans,
                                     size=self.al_config.budget,
                                     replace=False).tolist()

        self._update_data_pool(new_scans)

    def _acquisition_function(self):
        if self.epsilon != 0.1:
            new_eps = self.epsilon - 0.1
            self.epsilon = max(0.1, new_eps)
            self.main_logger.info(f'Epsilon changed to {self.epsilon}')

        if np.random.rand() < self.epsilon:
            self.main_logger.info('Random acquisition')
            self._random_acquisition()
        else:
            self.main_logger.info('Greedy acquisition')
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
