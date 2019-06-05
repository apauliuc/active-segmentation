import os

import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from scripts.predict import main_predict
from trainers.bayes_trainer import BayesianTrainer
from trainers.passive_trainer_ensemble import PassiveTrainerEnsemble
from trainers.passive_trainer import PassiveTrainer


def main_train_model(args, config_path: str):
    """Entry point for training a new model"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.gpu_node = args.gpu_node
    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    if config.training.bayesian is True:
        trainer = BayesianTrainer(config, run_dir)
    else:
        if config.training.use_ensemble:
            trainer = PassiveTrainerEnsemble(config, run_dir)
        else:
            trainer = PassiveTrainer(config, run_dir)

    trainer.run()

    if args.train_predict:
        main_predict(config, run_dir)
