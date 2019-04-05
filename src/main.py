import os
import yaml
import argparse

from alsegment.data.data_preprocess import preprocess_scans, separate_scans_to_slices
from alsegment.predict import prediction_main
from alsegment.trainer import Trainer
from alsegment.helpers.paths import get_new_run_path
from definitions import CONFIG_STANDARD, DATA_DIR_AT_AMC, DATA_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(args):
    if args.run_type == 'preprocess':
        data_root_dir = os.path.join('C:', 'Andrei', 'MHA and NPY')

        scan_names = preprocess_scans(data_root_dir, max_clip=255, clip_max_to_0=True)
        dir_test = os.path.join(DATA_DIR_AT_AMC, 'AMC Dummy')
        separate_scans_to_slices(data_root_dir, dir_test, scan_names, dummy_dataset=True)

    elif args.run_type == 'train':
        with open(args.config, 'r') as f:
            config = yaml.load(f)

        config['data']['path'] = args.ds_path
        run_dir = get_new_run_path(config['run_name'])

        with open(os.path.join(run_dir, 'cfg_file.yml'), 'w+') as f:
            yaml.dump(config, f)

        Trainer(config, run_dir).run()

        if args.train_predict:
            prediction_main(run_dir_name=run_dir, config=config, name=config['run_name'])

    elif args.run_type == 'predict':
        with open(os.path.join(args.run_dir, 'cfg_file.yml')) as f:
            config = yaml.load(f)

        prediction_main(args.run_dir, config)


if __name__ == '__main__':
    "Main starting point of the application"
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='train',
                        help='Type of run', choices=['train', 'predict', 'preprocess'])
    parser.add_argument('--config', type=str, default=CONFIG_STANDARD,
                        help='Configuration file to use')
    parser.add_argument('--ds_path', type=str, default=DATA_DIR_AT_AMC,
                        help='Path to main data directory')
    parser.add_argument('--train_predict', type=bool, default=True,
                        help='Indicate whether to predict after training is finished')
    parser.add_argument('--run_dir', type=str, default='',
                        help='Previous run directory to load model from (works only for predict run type)')

    arguments = parser.parse_args()
    main(arguments)