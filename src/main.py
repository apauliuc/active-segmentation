import os
import yaml
import argparse

from alsegment.trainer import Trainer
from alsegment.helpers.paths import get_new_run_path
from definitions import CONFIG_STANDARD, DATA_DIR_AT_AMC, DATA_DIR


def main(args):
    if args.run_type == 'train':
        with open(args.config, 'r') as f:
            config = yaml.load(f)

        config['data']['path'] = args.ds_path
        run_dir = get_new_run_path(config['run_name'])

        with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
            yaml.dump(config, f)

        Trainer(config, run_dir).run()

    elif args.run_type == 'predict':
        pass


if __name__ == '__main__':
    "Main starting point of the application"
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train',
                        help='Type of run', choices=['train', 'predict'])
    parser.add_argument('--config', type=str, default=CONFIG_STANDARD,
                        help='Configuration file to use')
    parser.add_argument('--ds_path', type=str, default=DATA_DIR,
                        help='Path to main data directory')

    arguments = parser.parse_args()
    main(arguments)
