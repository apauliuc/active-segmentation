import pickle
import shutil

import numpy as np
from os.path import join
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms as standard_transforms

from helpers.config import ConfigClass
from helpers.paths import recursive_glob_filenames


class ALMSRA10KPool(Dataset):
    """
        Data pool class for MSRA10K active learning
    """
    _data_pool: list
    _labelled_pool: list

    def __init__(self, config: ConfigClass):
        self.config = config
        self.al_config = config.active_learn
        self._input_size = (176, 160)

        self.data_root = join(config.data.path, config.data.dataset)
        self.train_path = join(self.data_root, 'train')

        self._data_pool = [x.split('.')[0] for x in recursive_glob_filenames(self.train_path, '.jpg')]

        self._create_initial_pool()

        with open(join(self.data_root, 'norm_data.pkl'), 'rb') as f:
            mean_std = pickle.load(f)

        self.input_transform = standard_transforms.Compose([
            standard_transforms.Resize(self._input_size),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(**mean_std)
        ])

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

    def copy_pool_files_to_dir(self, files: list, save_dir: str):
        for file in files:
            shutil.copy(join(self.train_path, f'{file}.jpg'), save_dir)

    def __len__(self):
        return len(self._data_pool)

    def __getitem__(self, index: int):
        image_name, _ = self._data_pool[index]
        image_path = join(self.train_path, f'{image_name}.jpg')

        image = Image.open(image_path).convert('RGB')

        image = self.input_transform(image)

        return image

    @property
    def train_pool(self):
        return self._labelled_pool

    @property
    def unlabelled_files(self):
        return self._data_pool
