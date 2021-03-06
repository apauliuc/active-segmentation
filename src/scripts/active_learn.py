import os
import yaml

from helpers.config import get_config_from_path
from helpers.paths import get_new_run_path
from scripts.evaluate_general import main_evaluation
from scripts.evaluate_mds import main_evaluation_mds
from trainers.active_learn_img.bald_combined_img import BALDCombinedImage
from trainers.active_learn_img.epsilon_greedy_img import EpsilonGreedyTrainerImage

from trainers.active_learn_scan.bald import BALDScan
from trainers.active_learn_scan.least_confident import LeastConfidentScan
from trainers.active_learn_scan.max_entropy import MaxEntropyScan
from trainers.active_learn_scan.random import RandomScan
from trainers.active_learn_scan.weighted_least_confident import WeightedLeastConfidentScan
from trainers.active_learn_scan.weighted_max_entropy import WeightedMaxEntropyScan
from trainers.active_learn_scan.bald_combined import BALDCombinedScan
from trainers.active_learn_scan.epsilon_greedy import EpsilonGreedyTrainerScan

from trainers.active_learn_img.bald_img import BALDImage
from trainers.active_learn_img.max_entropy_img import MaxEntropyImage
from trainers.active_learn_img.least_confident_img import LeastConfidentImage
from trainers.active_learn_img.random_img import RandomImage


def main_active_learning(args, config_path: str):
    """Entry point for training a new model with active learning"""
    config = get_config_from_path(config_path)

    config.data.mode = 'train'
    config.data.path = args.ds_path
    config.gpu_node = args.gpu_node
    config.training.loss_fn.gpu_node = args.gpu_node
    config.al_mode = True
    config.model.type = config.training.type

    config.active_learn.weighted = True if 'weighted' in config.active_learn.method else False

    if 'combined' in config.active_learn.method:
        config.prediction.mode = 'single'
        config.training.use_ensemble = True
        config.prediction.mc_passes = config.active_learn.mc_passes
        if 'skunet' not in config.model.name:
            config.model.network_params.dropout = True
    elif 'mc' in config.active_learn.method:
        config.prediction.mode = 'mc'
        config.prediction.mc_passes = config.active_learn.mc_passes
        config.training.use_ensemble = False
        if 'skunet' not in config.model.name:
            config.model.network_params.dropout = True
    elif 'ensemble' in config.active_learn.method:
        config.prediction.mode = 'single'
        config.training.use_ensemble = True
        config.model.network_params.dropout = False

    mds_flag = True if 'AMC' in config.data.dataset else False

    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    trainer_class = _get_al_trainer(config.active_learn.method, mds_flag)

    trainer = trainer_class(config, run_dir)
    trainer.run()

    if args.train_predict:
        if mds_flag:
            main_evaluation_mds(config, run_dir)
        else:
            main_evaluation(config, run_dir)


def _get_al_trainer(name_: str, mds_flag_: bool):
    try:
        if mds_flag_:
            return {
                'random': RandomScan,
                'least_confident_mc': LeastConfidentScan,
                'least_confident_ensemble': LeastConfidentScan,
                'max_entropy_mc': MaxEntropyScan,
                'max_entropy_ensemble': MaxEntropyScan,
                'bald_mc': BALDScan,
                'bald_ensemble': BALDScan,
                'weighted_least_confident_mc': WeightedLeastConfidentScan,
                'weighted_max_entropy_mc': WeightedMaxEntropyScan,
                'weighted_least_confident_ensemble': WeightedLeastConfidentScan,
                'weighted_max_entropy_ensemble': WeightedMaxEntropyScan,
                'bald_combined': BALDCombinedScan,
                'epsilon_greedy': EpsilonGreedyTrainerScan
            }[name_]
        else:
            return {
                'random': RandomImage,
                'least_confident_mc': LeastConfidentImage,
                'least_confident_ensemble': LeastConfidentImage,
                'max_entropy_mc': MaxEntropyImage,
                'max_entropy_ensemble': MaxEntropyImage,
                'bald_mc': BALDImage,
                'bald_ensemble': BALDImage,
                'bald_combined': BALDCombinedImage,
                'epsilon_greedy': EpsilonGreedyTrainerImage
            }[name_]
    except KeyError:
        raise Exception(f'Trainer {name_} not available')
