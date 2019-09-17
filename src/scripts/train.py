import os

import yaml

from helpers.config import get_config_from_path, ConfigClass
from helpers.paths import get_new_run_path
from scripts.predict_mds import main_predict_mds
from trainers.bayes_trainer import BayesianTrainer
from trainers.passive_trainer_ensemble import PassiveTrainerEnsemble
from trainers.passive_trainer import PassiveTrainer
from trainers.variational_trainer import VariationalTrainer
from trainers.variational_trainer_no_recon import VariationalTrainerNoReconstruction


def main_train_model(args, config_path: str):
    """Entry point for training a new model"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.gpu_node = args.gpu_node
    config.training.loss_fn.gpu_node = args.gpu_node
    config.model.type = config.training.type
    config.training.reconstruction = False if 'no_recon' in config.model.name else True

    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    trainer_cls = _get_trainer_type(config.training)

    trainer = trainer_cls(config, run_dir)
    trainer.run()

    if args.train_predict and 'AMC' in config.data.dataset:
        main_predict_mds(config, run_dir)


def _get_trainer_type(train_cfg: ConfigClass):
    if train_cfg.type == 'bayesian':
        return BayesianTrainer
    elif train_cfg.type == 'standard':
        if train_cfg.use_ensemble:
            return PassiveTrainerEnsemble
        else:
            return PassiveTrainer
    elif train_cfg.type == 'variational':
        if train_cfg.reconstruction:
            return VariationalTrainer
        else:
            return VariationalTrainerNoReconstruction
