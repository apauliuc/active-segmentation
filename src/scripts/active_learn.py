import os

import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from scripts.predict import main_predict
from trainers.active_learning.bald import BALDScan
from trainers.active_learning.least_confident import LeastConfidentScan
from trainers.active_learning.max_entropy import MaxEntropyScan
from trainers.active_learning.random import RandomScan


def main_active_learning(args, config_path: str):
    """Entry point for training a new model with active learning"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.gpu_node = args.args.gpu_node
    config.al_mode = True

    if 'mc' in config.active_learn.method:
        config.prediction.mode = 'mc'
        config.prediction.mc_passes = config.active_learn.mc_passes
        config.training.use_ensemble = False
        config.model.network_params.dropout = True
    elif 'ensemble' in config.active_learn.method:
        config.prediction.mode = 'single'
        config.training.use_ensemble = True
        config.model.network_params.dropout = False

    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    trainer_class = _get_al_trainer(config.active_learn.method)

    trainer = trainer_class(config, run_dir)
    trainer.run()

    if args.train_predict:
        main_predict(config, run_dir)


def _get_al_trainer(name: str):
    try:
        return {
            'random': RandomScan,
            'least_confident_mc': LeastConfidentScan,
            'least_confident_ensemble': LeastConfidentScan,
            'max_entropy_mc': MaxEntropyScan,
            'max_entropy_ensemble': MaxEntropyScan,
            'bald_mc': BALDScan,
            'bald_ensemble': BALDScan
        }[name]
    except KeyError:
        raise Exception(f'Trainer {name} not available')
