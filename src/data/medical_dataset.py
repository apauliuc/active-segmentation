import os
import pickle
import numpy as np
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.base_loader import BaseLoader
from data.data_transforms import ToTensor
from data.data_transforms import Normalize
from helpers.config import ConfigClass
from helpers.paths import get_dataset_path


class MedicalScanDataset(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: path to folder containing images
        :param transform: optional transform to apply on samples
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'image')
        self.segment_dir = os.path.join(data_dir, 'segment')

        with open(join(data_dir, 'file_list.pkl'), 'rb') as f:
            self.file_list = pickle.load(f)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        img_name = self.file_list[item]

        image = Image.open(join(self.image_dir, img_name))
        segmentation = Image.open(join(self.segment_dir, img_name))

        image, segmentation = self.transform((image, segmentation))

        return image, segmentation


class MDSPrediction(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, file_path, transform=lambda x: x):
        self.scan = np.load(file_path)
        self.scan = transform(np.transpose(self.scan, (1, 2, 0))).type(torch.FloatTensor)
        self.scan = self.scan.unsqueeze(1)

        self.shape = self.scan.shape

    def __len__(self) -> int:
        return len(self.scan)

    def __getitem__(self, item: int):
        image = self.scan[item, :, :]
        return image


class MDSDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, shuffle=True):
        assert config.mode in ['train', 'predict']

        self.config = config
        self.input_channels = 1
        self.num_classes = 1

        data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(data_root, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self.train_transform = transforms.Compose([
            ToTensor(),
            Normalize(ds_statistics['mean'], ds_statistics['std'])
        ])

        self.predict_transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([ds_statistics['mean']], [ds_statistics['std']]),
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
            self.predict_path = get_dataset_path(config.path, config.dataset, 'predict')
            self.dir_list = [x for x in os.listdir(self.predict_path)
                             if os.path.isdir(os.path.join(self.predict_path, x))]
        else:
            raise Exception('Data loading mode not found')

    def get_predict_loader(self, file_name):
        predict_dataset = MDSPrediction(os.path.join(self.predict_path, file_name), self.predict_transf)

        return DataLoader(predict_dataset,
                          batch_size=self.config.batch_size_val,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available())
