import os
import argparse
import subprocess

import torch

from data.data_preprocess_mds import mds_separate_scans_to_slices, mds_preprocess_scans
from helpers.config import get_config_from_path
from scripts.active_learn import main_active_learning
from scripts.predict import main_predict
from scripts.train import main_train_model
from definitions import CONFIG_DEFAULT, CONFIG_AL, DATA_DIR, RUNS_DIR, CONFIG_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(args):
    if args.run_type == 'preprocess':
        # Some data preprocessing and writing slices to separate files
        data_root_dir = os.path.join('C:', 'Andrei', 'MHA and NPY')
        dataset_path = os.path.join(DATA_DIR, 'AMC New')

        scan_names = mds_preprocess_scans(data_root_dir, max_clip=255, clip_max_to_0=True)
        mds_separate_scans_to_slices(data_root_dir, dataset_path, scan_names, dummy_dataset=False)

    elif args.run_type == 'train':
        # Run training using 1 config file
        print(f'Using config {args.config}')
        main_train_model(args, os.path.join(args.configs_dir, args.config))

    elif args.run_type == 'train_all_configs':
        # Run training for multiple config files given file name patter
        configs_list = sorted([x for x in os.listdir(args.configs_dir) if args.configs_pattern in x])
        print(f'Running training on {len(configs_list)} configs')

        for config in configs_list:
            torch.cuda.empty_cache()
            try:
                # Start new process which goes on 'train' branch
                subprocess.run(['python', 'main.py', '-r', 'train', '-c', config])
            except Exception as e:
                print(f'Error: {e}')
                print('\nSomething went wrong. Moving to next config file\n')

    elif args.run_type == 'predict':
        # Run prediction on the val dataset using previous run directory
        run_dir = os.path.join(RUNS_DIR, args.run_dir)
        config = get_config_from_path(os.path.join(run_dir, 'cfg_file.yml'))

        main_predict(config, run_dir)

    elif args.run_type == 'active_learning':
        # Run Active Learning training algorithm
        print(f'Using config {args.config}')
        main_active_learning(args, os.path.join(args.configs_dir, args.config))

    else:
        raise ValueError('Run type not known')


if __name__ == '__main__':
    "Main starting point of the application"
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_type', type=str, default='active_learning',
                        help='Type of run',
                        choices=['train', 'predict', 'preprocess', 'train_all_configs', 'active_learning'])
    parser.add_argument('-c', '--config', type=str, default=CONFIG_AL,
                        help='Configuration file to use')
    parser.add_argument('--configs_dir', type=str, default=CONFIG_DIR,
                        help='Directory of all configurations to train on (run_type = train_all_configs)')
    parser.add_argument('--configs_pattern', type=str, default='config_',
                        help='File name pattern for config files (run_type = train_all_configs)')
    parser.add_argument('-ds', '--ds_path', type=str, default=DATA_DIR,
                        help='Path to main data directory')
    parser.add_argument('-tp', '--train_predict', type=bool, default=False,
                        help='Indicate whether to predict after training is finished')
    parser.add_argument('--run_dir', type=str, default='',
                        help='Previous run directory to load model from (works only for run_type = predict)')

    arguments = parser.parse_args()
    main(arguments)
