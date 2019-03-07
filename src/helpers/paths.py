import os
import time
import datetime

from definitions import DATA_DIR, RUNS_DIR

CHECKPOINT_NAME_TEMPLATE = 'Checkpoint_%m-%d_%H-%M-%S'


def get_dataset_path(dataset_path=DATA_DIR, dataset_name='AMC', dataset_type='train') -> str:
    return os.path.join(dataset_path, dataset_name, dataset_type)


def create_checkpoint_name(time_signature: float) -> str:
    return datetime.datetime.fromtimestamp(time_signature).strftime(CHECKPOINT_NAME_TEMPLATE)


def get_new_checkpoint_path() -> str:
    path = os.path.join(RUNS_DIR, create_checkpoint_name(time.time()))
    create_directory(path)
    return path


def create_directory(path: str):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
