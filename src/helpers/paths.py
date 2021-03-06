import os
import time
import datetime

from definitions import DATA_DIR, RUNS_DIR

DEFAULT_NAME = 'Run'
NAME_TEMPLATE = '_%m-%d_%H-%M-%S'


def get_dataset_path(dataset_path=DATA_DIR, dataset_name='AMC', dataset_type='train') -> str:
    return os.path.join(dataset_path, dataset_name, dataset_type)


def create_run_name(name: str, time_signature: float) -> str:
    return name + datetime.datetime.fromtimestamp(time_signature).strftime(NAME_TEMPLATE)


def get_new_run_path(name: str) -> str:
    if name is None:
        name = DEFAULT_NAME
    path = os.path.join(RUNS_DIR, create_run_name(name, time.time()))
    create_directory(path)
    return path


def get_resume_model_path(run_dir: str, model_filename: str) -> str:
    if model_filename[-4:] != '.pth':
        model_filename = f'{model_filename}.pth'
    model_path = os.path.join(RUNS_DIR, run_dir, model_filename)
    return model_path


def get_resume_optimizer_path(run_dir: str, optim_filename: str) -> str:
    optim_path = os.path.join(RUNS_DIR, run_dir, f'{optim_filename}.pth')
    return optim_path


def create_directory(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def recursive_glob_filenames(root_dir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param root_dir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        filename
        for loop_root, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.endswith(suffix)
        and not filename.startswith('._')
    ]


def recursive_glob_paths(root_dir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param root_dir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(loop_root, filename)
        for loop_root, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
