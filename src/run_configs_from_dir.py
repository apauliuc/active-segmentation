import os
import shutil
import yaml

from train import train
from alsegment.helpers.paths import get_new_run_path
from definitions import DATA_DIR_AT_AMC, CONFIG_DIR


if __name__ == '__main__':
    for cfg_file in sorted(os.listdir(CONFIG_DIR)):
        if 'unet_' in cfg_file:
            cfg_path = os.path.join(CONFIG_DIR, cfg_file)

            with open(cfg_path) as f:
                config = yaml.load(f)

            # Define data directory from args
            config['data']['path'] = DATA_DIR_AT_AMC

            # Create logger, writer
            logging_dir = get_new_run_path(config['run_name'])
            shutil.copy(cfg_path, logging_dir)

            print(config)
            train(config, logging_dir)
