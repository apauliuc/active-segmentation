import os
import yaml
import argparse

from data.data_preprocess_mds import mds_separate_scans_to_slices, mds_preprocess_scans
from helpers.config import ConfigClass
from main_scripts.predict import prediction_main
from main_scripts.trainer import Trainer
from helpers.paths import get_new_run_path
from definitions import CONFIG_STANDARD, DATA_DIR, RUNS_DIR, CONFIG_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train_config(args, config_path: str):
    with open(config_path, 'r') as f:
        config = ConfigClass(yaml.load(f))

    config.data.mode = 'train'
    config.data.path = args.ds_path
    run_dir = get_new_run_path(config.run_name)

    with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
        yaml.dump(config, f)

    trainer = Trainer(config, run_dir)
    trainer.run()

    if args.train_predict:
        config.data.mode = 'predict'
        prediction_main(run_dir_name=run_dir, config=config, name=config.run_name)


def main(args):
    if args.run_type == 'preprocess':
        data_root_dir = os.path.join('C:', 'Andrei', 'MHA and NPY')
        dataset_path = os.path.join(DATA_DIR, 'AMC Dummy')

        scan_names = mds_preprocess_scans(data_root_dir, max_clip=255, clip_max_to_0=False)
        mds_separate_scans_to_slices(data_root_dir, dataset_path, scan_names, dummy_dataset=True)

    elif args.run_type == 'train':
        train_config(args, args.config)

    elif args.run_type == 'train_all_configs':
        configs_list = [os.path.join(args.configs_dir, x) for x in os.listdir(args.configs_dir)]

        for config in configs_list:
            train_config(args, config)

    elif args.run_type == 'predict':
        with open(os.path.join(RUNS_DIR, args.run_dir, 'cfg_file.yml')) as f:
            config = ConfigClass(yaml.load(f))
        config.data.mode = 'predict'

        prediction_main(args.run_dir, config, name=config.run_name)


if __name__ == '__main__':
    "Main starting point of the application"
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train_all_configs',
                        help='Type of run', choices=['train', 'predict', 'preprocess', 'train_all_configs'])
    parser.add_argument('--config', type=str, default=CONFIG_STANDARD,
                        help='Configuration file to use')
    parser.add_argument('--configs_dir', type=str, default=CONFIG_DIR,
                        help='Directory of all configurations to train on')
    parser.add_argument('--ds_path', type=str, default=DATA_DIR,
                        help='Path to main data directory')
    parser.add_argument('--train_predict', type=bool, default=True,
                        help='Indicate whether to predict after training is finished')
    parser.add_argument('--run_dir', type=str, default='',
                        help='Previous run directory to load model from (works only for run_type = predict)')

    arguments = parser.parse_args()
    main(arguments)
