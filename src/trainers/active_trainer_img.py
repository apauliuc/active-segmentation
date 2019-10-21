import os
import pickle

import numpy as np
from scipy.special import xlogy
import torch

from ignite.engine.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from helpers.torch_utils import apply_dropout
from trainers.base_trainer import BaseTrainer
from data import get_dataloaders
from data.base_loader import BaseLoader
from alsegment import get_pool_class
from helpers.config import ConfigClass
from helpers.utils import setup_logger
from alsegment.msra10k_pool import ALMSRA10KPool


class ActiveTrainer(BaseTrainer):
    """
    Base implementation of AL trainer for single files
    """
    data_loaders: BaseLoader
    acquisition_step: int

    def __init__(self, config: ConfigClass, save_dir: str, name='ActiveTrainer'):
        super(ActiveTrainer, self).__init__(config, save_dir, name)

        self.main_data_dir = os.path.join(self.save_dir, 'Datasets')
        os.makedirs(self.main_data_dir)

        self.al_config = config.active_learn

        self.data_pool = get_pool_class(self.config)
        self.data_loaders = get_dataloaders(self.config.data, file_list=self.data_pool.train_pool)
        self.main_logger.info(self.data_loaders.msg)

        self._create_train_loggers(value=0)
        self._save_dataset_info()
        self.data_pool.copy_pool_files_to_dir(self.data_pool.train_pool, self.save_data_dir)

        if self.use_ensemble:
            self._init_train_components_ensemble()
        else:
            self._init_train_components()

    def _create_train_loggers(self, value):
        self.acquisition_step = value
        self.save_model_dir = os.path.join(self.save_dir, f'Step {value}')
        os.makedirs(self.save_model_dir)
        self.save_data_dir = os.path.join(self.main_data_dir, f'Step {value}')
        os.makedirs(self.save_data_dir)

        self.train_logger, self.train_log_handler = setup_logger(self.save_model_dir, f'Train step {value}')
        self.train_writer = SummaryWriter(log_dir=self.save_model_dir)

    def _save_dataset_info(self):
        save_dict = {
            'labelled_pool': self.data_pool.train_pool,
            'data_pool': self.data_pool.unlabelled_files
        }

        with open(os.path.join(self.save_model_dir, 'dataset_info.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)

        with open(os.path.join(self.save_model_dir, 'data_pool.pkl'), 'wb') as f:
            pickle.dump(self.data_pool, f)

    def _update_components(self):
        self.train_cfg.num_epochs += 5
        # Recreate components
        if self.use_ensemble:
            self._init_train_components_ensemble(reinitialise=True)
        else:
            self._init_train_components(reinitialise=True)

    def _on_epoch_completed(self, _engine: Engine) -> None:
        self._log_training_results(_engine, self.train_logger, self.train_writer)
        self._evaluate_on_val(_engine, self.train_logger, self.train_writer)

    def _init_handlers(self, _init_checkpoint=True) -> None:
        self._init_checkpoint_handler(save_dir=self.save_model_dir)
        super()._init_handlers(_init_checkpoint=False)

    def _finalize(self, on_error=False) -> None:
        if not on_error:
            if self.trainer.should_terminate:
                self.train_logger.info(f'Early stopping on epoch {self.trainer.state.epoch}')

            # Evaluate model and save information
            eval_loss = self.evaluator.state.metrics['loss']
            eval_metrics = self.evaluator.state.metrics['segment_metrics']

            msg = f'Step {self.acquisition_step} - After {self.trainer.state.epoch} training epochs: ' \
                  f'Val. loss: {eval_loss:.4f}   ' \
                  f'IoU: {eval_metrics["avg_iou"]:.4f}   ' \
                  f'F1: {eval_metrics["avg_f1"]:.4f}   ' \
                  f'mAP: {eval_metrics["mAP"]}'
            self.main_logger.info(msg)

            self.main_writer.add_scalar(f'active_learning/avg_val_loss', eval_loss, self.acquisition_step)
            for key, value in eval_metrics.items():
                self.main_writer.add_scalar(f'active_learning/{key}', value, self.acquisition_step)
            self.main_writer.add_scalar(f'active_learning/epochs_trained', self.trainer.state.epoch,
                                        self.acquisition_step)

        # Close writer and logger related to model training
        self.train_writer.close()
        self.train_logger.removeHandler(self.train_log_handler)

    def _finalize_trainer(self) -> None:
        # Close writer and logger related to trainer class
        self.main_writer.close()
        self.main_logger.removeHandler(self.main_log_handler)

    def _train(self) -> None:
        self.trainer.run(self.data_loaders.train_loader, max_epochs=self.train_cfg.num_epochs)

    def run(self) -> None:
        self.main_logger.info(f'{self.log_name} initialised. Starting training on {self.device}.')
        self.main_logger.info(f'Active_Training - acquisition step 0')
        self.main_logger.info(f'Using {len(self.data_pool.train_pool)} datapoints.')
        self._train()

        for step in range(1, self.al_config.acquisition_steps + 1):
            if len(self.data_pool.unlabelled_files) < self.al_config.budget:
                self.main_logger.info(f'Data pool too small. Stopping training.')
                break

            self.main_logger.info(f'Active_Training - acquisition step {step}')

            self._create_train_loggers(value=step)

            self._acquisition_function()

            self._save_dataset_info()

            self._update_components()

            self.main_logger.info(f'Using {len(self.data_pool.train_pool)} datapoints.')
            self._train()

        self._finalize_trainer()

    def _update_data_pool(self, new_data_points: list):
        self.data_pool.update_train_pool(new_data_points)
        self.data_loaders.update_train_loader(self.data_pool.train_pool)

        self.data_pool.copy_pool_files_to_dir(new_data_points, self.save_data_dir)

    def _acquisition_function(self) -> None:
        pass

    def _predict_proba(self, m_type='mc_dropout'):
        assert m_type in ['mc_dropout', 'ensemble']

        al_loader = DataLoader(self.data_pool,
                               batch_size=self.config.data.batch_size_val,
                               shuffle=False,
                               num_workers=self.config.data.num_workers,
                               pin_memory=torch.cuda.is_available())
        predictions = []

        if m_type == 'mc_dropout':
            self.model.eval()
            self.model.apply(apply_dropout)
            size_pred = self.al_config.mc_passes
        else:
            for model in self.ens_models:
                model.eval()
            size_pred = len(self.ens_models)

        with torch.no_grad():
            for batch in al_loader:
                x = batch
                x = x.to(self.device)

                out_probas = torch.zeros_like(x)

                if m_type == 'mc_dropout':
                    for i in range(self.al_config.mc_passes):
                        out = self.model(x)
                        out_probas += torch.sigmoid(out)
                else:
                    for model in self.ens_models:
                        out = model(x)
                        out_probas += torch.sigmoid(out)

                out_probas = out_probas.reshape((out_probas.shape[0], -1)).cpu()
                out_probas = out_probas / size_pred

                predictions.extend(*out_probas.split(out_probas.shape[0]))

        return torch.stack(predictions)

    def _predict_proba_individual(self, m_type='mc_dropout'):
        assert m_type in ['mc_dropout', 'ensemble']

        al_loader = DataLoader(self.data_pool,
                               batch_size=self.config.data.batch_size_val,
                               shuffle=False,
                               num_workers=self.config.data.num_workers,
                               pin_memory=torch.cuda.is_available())

        predictions = list()

        if m_type == 'mc_dropout':
            self.model.eval()
            self.model.apply(apply_dropout)
        else:
            for model in self.ens_models:
                model.eval()

        with torch.no_grad():
            for batch in al_loader:
                x = batch
                x = x.to(self.device)

                if m_type == 'mc_dropout':
                    count_iter = self.al_config.mc_passes
                else:
                    count_iter = len(self.ens_models)

                for i in range(count_iter):
                    if m_type == 'mc_dropout':
                        out = self.model(x)
                    else:
                        out = self.ens_models[i](x)

                    out_probas = torch.sigmoid(out).reshape((out.shape[0], -1)).cpu()

                    if len(predictions) < i + 1:
                        predictions.append(out_probas)
                    else:
                        predictions[i] = torch.cat((predictions[i], out_probas))

        return [pred.cpu() for pred in predictions]

    @staticmethod
    def _compute_pixel_entropy(x):
        proba = np.expand_dims(x, 0)
        p = np.concatenate([proba, 1 - proba])
        logp = xlogy(np.sign(p), p) / np.log(2)
        return -np.nansum(p * logp, axis=0)
