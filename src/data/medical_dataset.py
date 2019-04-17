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


class MDSMain(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, data_dir, file_list=None, transform=lambda x: x):
        """
        :param data_dir: path to folder containing images
        :param file_list: list of train files to use (optional, used for active learning)
        :param transform: optional transform to apply on samples
        """
        if file_list is None:
            with open(join(data_dir, 'file_list.pkl'), 'rb') as f:
                file_list = pickle.load(f)

        self.file_list = file_list

        self.image_dir = os.path.join(data_dir, 'image')
        self.segment_dir = os.path.join(data_dir, 'segment')

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
    """Prediction Medical Scans Dataset"""

    def __init__(self, predict_path, dir_name, in_transform=lambda x: x, out_transform=lambda x: x):
        dir_path = os.path.join(predict_path, dir_name)
        in_name = os.path.join(dir_path, f'{dir_name}_scan.npy')
        out_name = os.path.join(dir_path, f'{dir_name}_segmentation.npy')

        self.scan = np.load(in_name)
        self.scan = in_transform(np.transpose(self.scan, (1, 2, 0))).type(torch.FloatTensor)
        self.scan = self.scan.unsqueeze(1)

        self.segment = np.load(out_name)
        self.segment = out_transform(np.transpose(self.segment, (1, 2, 0))).type(torch.FloatTensor)
        self.segment = self.segment.unsqueeze(1)

        self.shape = self.scan.shape

    def __len__(self) -> int:
        return len(self.scan)

    def __getitem__(self, item: int):
        image = self.scan[item, :, :]
        segment = self.segment[item, :, :]
        return image, segment


class MDSDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, file_list=None, shuffle=True):
        assert config.mode in ['train', 'predict']

        self.config = config
        self.shuffle = shuffle
        self.input_channels = 1
        self.num_classes = 1

        data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(data_root, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self.train_transform = transforms.Compose([
            ToTensor(),
            Normalize(ds_statistics['mean'], ds_statistics['std'])
        ])

        self.predict_in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float().div(255)),
            transforms.Normalize([ds_statistics['mean']], [ds_statistics['std']]),
        ])

        self.predict_out_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float().div(255))
        ])

        if config.mode == 'train':
            self.train_path = get_dataset_path(config.path, config.dataset, 'train')
            val_path = get_dataset_path(config.path, config.dataset, 'val')

            train_dataset = MDSMain(self.train_path, file_list=file_list, transform=self.train_transform)
            val_dataset = MDSMain(val_path, transform=self.train_transform)

            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                                           num_workers=config.num_workers,
                                           pin_memory=torch.cuda.is_available())

            self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=shuffle,
                                         num_workers=config.num_workers,
                                         pin_memory=torch.cuda.is_available())

            if file_list is None:
                self.msg = f'Train data loader created from {self.train_path}.' \
                    f'Validation data loader created from {val_path}'
            else:
                self.msg = f'AL train data loader created from {self.train_path}.' \
                    f'Validation data loader created from {val_path}'

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

    def get_predict_loader(self, dir_name) -> DataLoader:
        predict_dataset = MDSPrediction(self.predict_path, dir_name,
                                        in_transform=self.predict_in_transform,
                                        out_transform=self.predict_out_transform)

        return DataLoader(predict_dataset,
                          batch_size=self.config.batch_size_val,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available())

    def update_train_loader(self, new_file_list: list) -> None:
        new_train_dataset = MDSMain(self.train_path, file_list=new_file_list, transform=self.train_transform)

        self.train_loader = DataLoader(new_train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.config.num_workers,
                                       pin_memory=torch.cuda.is_available())
