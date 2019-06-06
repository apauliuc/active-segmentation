import pickle
import shutil

import numpy as np
from os.path import join
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from helpers.config import ConfigClass
from helpers.paths import get_dataset_path


class ALPatientPool:
    """
    Data pool class for patient based active learning
    """
    _scans_pool: list
    _files_pool: list
    _labelled_scans_pool: list
    _labelled_files_pool: list

    def __init__(self, config: ConfigClass):
        self.config = config
        self.al_config = config.active_learn

        train_path = get_dataset_path(config.data.path, config.data.dataset, 'train')
        self.image_path = join(train_path, 'image')

        with open(join(train_path, 'file_list.pkl'), 'rb') as f:
            self._files_pool = pickle.load(f)

        self._scans_pool = list(set([x.split('_')[0] for x in self._files_pool]))
        self._create_initial_pool()

        # Transforms
        with open(join(config.data.path, config.data.dataset, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([ds_statistics['mean']], [ds_statistics['std']]),
        ])

    def _remove_from_pool(self, scans_to_remove: list) -> None:
        try:
            files_to_remove = [x for x in self._files_pool if x.split('_')[0] in scans_to_remove]
            for item in scans_to_remove:
                self._scans_pool.remove(item)
            for file in files_to_remove:
                self._files_pool.remove(file)
        except ValueError:
            pass

    def _create_initial_pool(self) -> None:
        if len(self.al_config.init_scans) > 0:
            init_scans_pool = self.al_config.init_scans
        else:
            if self.al_config.init_pool_size >= len(self._scans_pool):
                print('Initial pool size too large. Setting it to default 1')
                self.al_config.init_pool_size = 1

            init_scans_pool = np.random.choice(self._scans_pool,
                                               size=self.al_config.init_pool_size,
                                               replace=False).tolist()

        self._labelled_scans_pool = init_scans_pool
        self._labelled_files_pool = [x for x in self._files_pool if x.split('_')[0] in init_scans_pool]

        self._remove_from_pool(init_scans_pool)

    def update_train_pool(self, scans_list: list) -> None:
        self._labelled_scans_pool.extend(scans_list)
        new_files_list = [x for x in self._files_pool if x.split('_')[0] in scans_list]
        self._labelled_files_pool.extend(new_files_list)

        self._remove_from_pool(scans_list)

    def copy_pool_scans_to_dir(self, scan_ids: list, save_dir: str):
        files_to_copy = [x for x in self._labelled_files_pool if x.split('_')[0] in scan_ids]
        for file in files_to_copy:
            shutil.copy(join(self.image_path, file), save_dir)

    def get_files_list_for_scan(self, scan_id):
        return [x for x in self._files_pool if scan_id in x]

    @property
    def unlabelled_scans(self):
        return self._scans_pool

    @property
    def unlabelled_files(self):
        return self._files_pool

    @property
    def labelled_scans(self):
        return self._labelled_scans_pool

    @property
    def train_pool(self):
        return self._labelled_files_pool


class ALPatientDataset(Dataset):
    def __init__(self, files_list, image_path, in_transform):
        self._files_list = files_list
        self._image_path = image_path
        self._input_transform = in_transform

    def __len__(self):
        return len(self._files_list)

    def __getitem__(self, index: int):
        img_name = self._files_list[index]

        image = Image.open(join(self._image_path, img_name))

        image = self._input_transform(image)

        return image, index
