import math
import torch
from ignite import metrics
from ignite.engine.engine import Engine

from bayesian.layers import GaussianVariationalInference
from data import get_dataloaders
from helpers.config import ConfigClass
from helpers.metrics import SegmentationMetrics
from losses import get_loss_function
from trainers.base_trainer import BaseTrainer


class BayesianTrainer(BaseTrainer):

    def __init__(self, config: ConfigClass, save_dir: str):
        super(BayesianTrainer, self).__init__(config, save_dir, 'BBB_Trainer')

        self.data_loaders = get_dataloaders(config.data)
        self.main_logger.info(self.data_loaders.msg)

        self._init_train_components()

    def _init_train_components(self):
        self.metrics = {
            'loss': metrics.Loss(GaussianVariationalInference(get_loss_function(self.train_cfg.loss_fn))),
            'segment_metrics': SegmentationMetrics(num_classes=self.data_loaders.num_classes,
                                                   threshold=self.config.binarize_threshold)
        }

        self.model_cfg.network_params.input_channels = self.data_loaders.input_channels
        self.model_cfg.network_params.num_classes = self.data_loaders.num_classes

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.vi = GaussianVariationalInference(self._init_criterion())

        self.lr_scheduler = self._init_lr_scheduler(self.optimizer)

        self.trainer, self.evaluator = self._init_engines()

        self._init_handlers()

    def _init_engines(self):
        trainer = self._init_bayesian_trainer_engine(self.model, self.optimizer, self.vi, self.device)
        evaluator = self._init_bayesian_evaluator_engine(self.model, self.metrics, self.device)

        metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')

        return trainer, evaluator

    def _init_bayesian_trainer_engine(self, model, optimizer, vi, device) -> Engine:
        m = math.ceil(len(self.data_loaders.train_dataset) / self.config.data.batch_size)

        def _update(engine_, batch):
            model.train()
            x, y = batch
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            if self.train_cfg.beta_type == "Blundell":
                beta = 2 ** (m - (engine_.state.iteration + 1)) / (2 ** m - 1)
            elif self.train_cfg.beta_type == "Soenderby":
                beta = min(engine_.state.epoch / (self.train_cfg.num_epochs // 4), 1)
            elif self.train_cfg.beta_type == "Standard":
                beta = 1 / m
            else:
                beta = 0

            optimizer.zero_grad()
            outputs, kl = model.probforward(x)
            loss = vi(outputs, y, kl, beta)
            loss.backward()
            optimizer.step()
            return loss.item()

        return Engine(_update)

    def _init_bayesian_evaluator_engine(self, model, eval_metrics, device) -> Engine:
        m = math.ceil(len(self.data_loaders.val_dataset) / self.config.data.batch_size_val)

        def _inference(engine_, batch):
            model.eval()

            with torch.no_grad():
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if self.train_cfg.beta_type == "Blundell":
                    beta = 2 ** (m - (engine_.state.iteration + 1)) / (2 ** m - 1)
                elif self.train_cfg.beta_type == "Soenderby":
                    beta = min(self.trainer.state.epoch / (self.train_cfg.num_epochs // 4), 1)
                elif self.train_cfg.beta_type == "Standard":
                    beta = 1 / m
                else:
                    beta = 0

                outputs, kl = model.probforward(x)

                return outputs, y, {'kl': kl, 'beta': beta}

        engine = Engine(_inference)

        for name, metric in eval_metrics.items():
            metric.attach(engine, name)

        return engine
