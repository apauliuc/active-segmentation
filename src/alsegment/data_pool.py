import os
import pickle
import numpy as np

from helpers.config import ConfigClass
from helpers.paths import get_dataset_path


class ALDataPool:
    _data_pool: list
    _labelled_pool: list

    def __init__(self, config: ConfigClass):
        self.config = config
        self.al_config = config.active_learn
        self._train_path = get_dataset_path(config.path, config.dataset, 'train')

        with open(os.path.join(self._train_path, 'file_list.pkl'), 'rb') as f:
            self._data_pool = pickle.load(f)

    def _remove_list_from_data_pool(self, to_remove: list):
        for item in to_remove:
            self._data_pool.remove(item)
        # try:
        #     for item in to_remove:
        #         self._data_pool.remove(item)
        # except ValueError:
        #     pass

    def create_initial_pool(self):
        init_pool = np.random.choice(self._data_pool,
                                     size=self.al_config.init_pool_size,
                                     replace=False).tolist()
        self._labelled_pool.extend(init_pool)
        self._remove_list_from_data_pool(init_pool)

    def update_train_pool(self, item_list: list):
        self._labelled_pool.extend(item_list)
        self._remove_list_from_data_pool(item_list)

    @property
    def train_pool(self):
        return self._labelled_pool

    @property
    def train_pool_len(self):
        return len(self._labelled_pool)

    @property
    def data_pool(self):
        return self._data_pool

    @property
    def data_pool_len(self):
        return len(self._data_pool)
