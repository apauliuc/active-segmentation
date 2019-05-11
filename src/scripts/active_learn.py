import os

import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from scripts.predict import main_predict
from trainers.al_scan_trainers.random_scan import RandomScan
from trainers.al_scan_trainers.least_confident_scan import LeastConfidentMCScan
from trainers.al_trainers.random import Random
from trainers.al_trainers.least_confident import LeastConfident


def main_active_learning(args, config_path: str):
    """Entry point for training a new model with active learning"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.al_mode = True

    if 'mc' in config.active_learn.method:
        config.prediction.mode = 'mc'
        config.prediction.mc_passes = config.active_learn.mc_passes

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
            'random': Random,
            'least_confident': LeastConfident,
            'least_confident_mc': LeastConfident,
            'random_scan': RandomScan,
            'least_confident_mc_scan': LeastConfidentMCScan
        }[name]
    except KeyError:
        raise Exception(f'Trainer {name} not available')
