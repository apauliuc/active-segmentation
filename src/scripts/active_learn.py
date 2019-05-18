import os

import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from scripts.predict import main_predict
from trainers.active_learning.bald_mc import BALDScanMC
from trainers.active_learning.max_entropy_ensemble import MaxEntropyScanEnsemble
from trainers.active_learning.max_entropy_mc import MaxEntropyScanMC
from trainers.active_learning.random import RandomScan
from trainers.active_learning.least_confident_mc import LeastConfidentScanMC
from trainers.al_trainers.random_img import Random
from trainers.al_trainers.least_confident_img import LeastConfident


def main_active_learning(args, config_path: str):
    """Entry point for training a new model with active learning"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.al_mode = True

    if 'mc' in config.active_learn.method:
        config.prediction.mode = 'mc'
        config.prediction.mc_passes = config.active_learn.mc_passes
        config.training.use_ensemble = False
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
            # 'random': Random,
            # 'least_confident': LeastConfident,
            # 'least_confident_mc': LeastConfident,
            'random_scan': RandomScan,
            'least_confident_mc_scan': LeastConfidentScanMC,
            'max_entropy_mc': MaxEntropyScanMC,
            'max_entropy_ensemble': MaxEntropyScanEnsemble,
            'bald_mc': BALDScanMC
        }[name]
    except KeyError:
        raise Exception(f'Trainer {name} not available')
