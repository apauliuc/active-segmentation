import shutil

import numpy as np
from os.path import join
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms as standard_transforms

from data.cityscapes import get_cityscapes_files
from helpers.config import ConfigClass


class ALCityScapesPool(Dataset):
    """
        Data pool class for patient based active learning
    """
    _data_pool: list
    _labelled_pool: list

    def __init__(self, config: ConfigClass):
        self.config = config
        self.al_config = config.active_learn

        # self.data_root = os.path.join(config.path, config.dataset)
        self.data_root = '/Users/andrei/Programming/CityscapesDataset/'

        self._data_pool, self._image_path = get_cityscapes_files(self.data_root, 'train')

        mean_std = ([0.3006, 0.3365, 0.2956], [0.1951, 0.1972, 0.1968])
        self.input_transform = standard_transforms.Compose([
            standard_transforms.Resize((1024, 512)),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
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

    def copy_pool_files_to_dir(self, files: list, save_dir: str):
        for file in files:
            shutil.copy(join(self._image_path, file), save_dir)

    def __len__(self):
        return len(self._data_pool)

    def __getitem__(self, index: int):
        image_path, _ = self._data_pool[index]

        image = Image.open(image_path).convert('RGB')

        image = self.input_transform(image)

        return image, index

    @property
    def train_pool(self):
        return self._labelled_pool

    @property
    def unlabelled_files(self):
        return self._data_pool
