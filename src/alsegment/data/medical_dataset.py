import os
import pickle
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from alsegment.data.base_loader import BaseLoader
from alsegment.data.data_transforms import ToPILImage
from alsegment.data.data_transforms import ToTensor
from alsegment.data.data_transforms import Normalize
from alsegment.helpers.config import ConfigClass
from alsegment.helpers.paths import get_dataset_path


class MedicalScanDataset(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: path to folder containing images
        :param transform: optional transform to apply on samples
        """
        self.n_channels = 1
        self.n_classes = 1

        self.data_dir = data_dir

        with open(join(data_dir, 'file_list.pkl'), 'rb') as f:
            self.file_list = pickle.load(f)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        img_name = self.file_list[item]

        image = np.load(join(self.data_dir, img_name + '.npy'))
        segmentation = np.load(join(self.data_dir, img_name + '_seg.npy'))

        image, segmentation = self.transform((image, segmentation))

        return image, segmentation


class MDSDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, shuffle=True):
        assert config.mode in ['train', 'predict']

        self.input_channels = 1
        self.num_classes = 1

        data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(data_root, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self.train_transform = transforms.Compose([
            ToPILImage(),
            ToTensor(),
            Normalize(ds_statistics['mean'], ds_statistics['std'])
        ])

        if config.mode == 'train':
            train_path = get_dataset_path(config.path, config.dataset, 'train')
            val_path = get_dataset_path(config.path, config.dataset, 'val')

            train_dataset = MedicalScanDataset(train_path, self.train_transform)
            val_dataset = MedicalScanDataset(val_path, self.train_transform)

            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                                           num_workers=config.num_workers,
                                           pin_memory=torch.cuda.is_available())

            self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=shuffle,
                                         num_workers=config.num_workers,
                                         pin_memory=torch.cuda.is_available())

            self.train_iterations = (len(train_dataset) + config.batch_size) // config.batch_size
            self.val_iterations = (len(val_dataset) + config.batch_size) // config.batch_size

            self.msg = f'Train data loader created from {train_path}. Validation data loader created from {val_path}'

            if config.run_val_on_train:
                self.val_train_loader = DataLoader(train_dataset, batch_size=config.batch_size_val,
                                                   shuffle=shuffle, num_workers=config.num_workers,
                                                   pin_memory=torch.cuda.is_available())

        elif config.mode == 'predict':
            pass
        else:
            raise Exception('Data loading mode not found')
