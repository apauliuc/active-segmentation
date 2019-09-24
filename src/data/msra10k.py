import os
import pickle
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from data.base_loader import BaseLoader
from helpers.config import ConfigClass
from helpers.paths import recursive_glob_filenames
import data.cityscapes_transforms as custom_transforms
import torchvision.transforms as standard_transforms


class MSRA10KDataset(Dataset):
    """MSRA10K Dataset"""

    def __init__(self, data_dir, split, file_list=None, dataset_stats=None, input_size=(176, 160)):
        self.split = split
        self.input_size = input_size
        self.root_dir = join(data_dir, split)

        if file_list is None:
            file_list = [x.split('.')[0] for x in recursive_glob_filenames(self.root_dir, '.jpg')]

        self.file_list = file_list

        self.joint_transform, self.input_transform, self.target_transform = self._get_transforms(dataset_stats)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        file_name = self.file_list[item]

        image = Image.open(join(self.root_dir, f'{file_name}.jpg'))
        segmentation = Image.open(join(self.root_dir, f'{file_name}.png'))

        if self.joint_transform is not None:
            image, segmentation = self.joint_transform(image, segmentation)

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            segmentation = self.target_transform(segmentation)

        return image, segmentation

    def get_image_from_name(self, name: str):
        image = Image.open(join(self.root_dir, f'{name}.jpg'))
        segmentation = Image.open(join(self.root_dir, f'{name}.png'))

        if self.joint_transform is not None:
            image, segmentation = self.joint_transform(image, segmentation)

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            segmentation = self.target_transform(segmentation)

        return image, segmentation

    def _get_transforms(self, mean_std):
        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size)
                # custom_transforms.RandomHorizontalFlip(),
            ])
        elif self.split == 'val':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size)
                # custom_transforms.RandomHorizontalFlip(),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(**mean_std)
        ])

        target_transform = standard_transforms.Compose([
            standard_transforms.ToTensor()
        ])

        return joint, input_transform, target_transform


class MSRA10KDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, file_list=None, shuffle=True):

        self.config = config
        self.shuffle = shuffle
        self.input_channels = 3
        self.num_classes = 1
        self.image_size = (176, 160)  # width x height

        self.data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(self.data_root, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        if config.mode == 'train':
            self.train_dataset = MSRA10KDataset(self.data_root, 'train', file_list=file_list,
                                                dataset_stats=ds_statistics, input_size=self.image_size)
            self.val_dataset = MSRA10KDataset(self.data_root, 'val',
                                              dataset_stats=ds_statistics, input_size=self.image_size)

            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=shuffle,
                                           num_workers=config.num_workers,
                                           pin_memory=torch.cuda.is_available())

            self.val_loader = DataLoader(self.val_dataset,
                                         batch_size=config.batch_size_val,
                                         shuffle=shuffle,
                                         num_workers=config.num_workers,
                                         pin_memory=torch.cuda.is_available())

            if file_list is None:
                self.msg = f'Data loaders created from {self.data_root}'
            else:
                self.msg = f'AL train data loader created from {self.data_root}'

            if config.run_val_on_train:
                self.val_train_loader = DataLoader(self.train_dataset,
                                                   batch_size=config.batch_size_val,
                                                   shuffle=shuffle,
                                                   num_workers=config.num_workers,
                                                   pin_memory=torch.cuda.is_available())
        elif config.mode == 'evaluate':
            self.evaluation_dataset = MSRA10KDataset(self.data_root, 'val',
                                                     dataset_stats=ds_statistics, input_size=self.image_size)

            self.evaluation_loader = DataLoader(self.evaluation_dataset,
                                                batch_size=config.batch_size_val,
                                                shuffle=True,
                                                num_workers=config.num_workers,
                                                pin_memory=torch.cuda.is_available())
        else:
            raise Exception('Data loading mode not found')

    def update_train_loader(self, new_file_list: list) -> None:
        new_train_dataset = MSRA10KDataset(self.data_root, split='train', file_list=new_file_list,
                                           dataset_stats=self.ds_statistics)

        self.train_loader = DataLoader(new_train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.config.num_workers,
                                       pin_memory=torch.cuda.is_available())
