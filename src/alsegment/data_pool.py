import pickle
import numpy as np
from os.path import join
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from helpers.config import ConfigClass
from helpers.paths import get_dataset_path


class ALDataPool(Dataset):
    _data_pool: list
    _labelled_pool: list
    _name2idx: dict
    _idx2name: dict

    def __init__(self, config: ConfigClass):
        self.config = config
        self.al_config = config.active_learn

        self._train_path = get_dataset_path(config.data.path, config.data.dataset, 'train')

        with open(join(self._train_path, 'file_list.pkl'), 'rb') as f:
            self._data_pool = pickle.load(f)

        with open(join(config.data.path, config.data.dataset, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self._name2idx = {}
        self._idx2name = {}

        for idx, fname in enumerate(self._data_pool):
            self._name2idx[fname] = idx
            self._idx2name[idx] = fname

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([ds_statistics['mean']], [ds_statistics['std']]),
        ])

        self._create_initial_pool()

    def _remove_from_data_pool(self, to_remove: list) -> None:
        try:
            for item in to_remove:
                self._data_pool.remove(item)
        except ValueError:
            pass

    def _create_initial_pool(self) -> None:
        init_pool = np.random.choice(self._data_pool,
                                     size=self.al_config.init_pool_size,
                                     replace=False).tolist()
        self._labelled_pool = init_pool
        self._remove_from_data_pool(init_pool)

    def update_train_pool(self, item_list: list) -> None:
        self._labelled_pool.extend(item_list)
        self._remove_from_data_pool(item_list)

    def __len__(self):
        return len(self._data_pool)

    def __getitem__(self, index: int):
        img_name = self._data_pool[index]

        image = Image.open(join(self._train_path, img_name))

        image = self.input_transform(image)

        return image, index

    @property
    def train_pool(self):
        return self._labelled_pool

    @property
    def data_pool(self):
        return self._data_pool

    @property
    def name2idx(self):
        return self._name2idx

    @property
    def idx2name(self):
        return self._idx2name
