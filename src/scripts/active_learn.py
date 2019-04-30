import os

import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from trainers.active_trainer import ActiveTrainer
from trainers.least_confident_trainer import LeastConfidentTrainer
from trainers.least_confident_mc_trainer import LeastConfidentMonteCarloTrainer
from trainers.random_sample_trainer import RandomSampleTrainer


def main_active_learning(args, config_path: str):
    """Entry point for training a new model"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    al_method = config.active_learn.method

    if al_method == 'random':
        trainer = RandomSampleTrainer(config, run_dir)
    elif al_method == 'least_confident':
        trainer = LeastConfidentTrainer(config, run_dir)
    elif al_method == 'least_confident_mc':
        trainer = LeastConfidentMonteCarloTrainer(config, run_dir)
    else:
        trainer = ActiveTrainer(config, run_dir)

    trainer.run()

    # if args.train_predict:
    #     main_predict(config, run_dir)
