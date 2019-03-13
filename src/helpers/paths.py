import os
import time
import datetime

from definitions import DATA_DIR, RUNS_DIR

RUN_NAME_TEMPLATE = 'Run_%m-%d_%H-%M-%S'


def get_dataset_path(dataset_path=DATA_DIR, dataset_name='AMC', dataset_type='train') -> str:
    return os.path.join(dataset_path, dataset_name, dataset_type)


def create_run_name(time_signature: float) -> str:
    return datetime.datetime.fromtimestamp(time_signature).strftime(RUN_NAME_TEMPLATE)


def get_new_run_path() -> str:
    path = os.path.join(RUNS_DIR, create_run_name(time.time()))
    create_directory(path)
    return path


def get_model_optimizer_path(run_dir: str, model_filename: str, optim_filename: str) -> tuple:
    model_path = os.path.join(RUNS_DIR, run_dir, f'{model_filename}.pth')
    optim_path = os.path.join(RUNS_DIR, run_dir, f'{optim_filename}.pth')
    return model_path, optim_path


def create_directory(path: str):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
