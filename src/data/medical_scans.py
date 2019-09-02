import os
import pickle
import numpy as np
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.base_loader import BaseLoader
from helpers.config import ConfigClass
from helpers.paths import get_dataset_path
import data.medical_scans_transforms as custom_transforms
import torchvision.transforms as standard_transforms


# noinspection DuplicatedCode
class MDSMain(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, data_dir, split, file_list=None, dataset_stats=None):
        self.split = split

        if file_list is None:
            with open(join(data_dir, 'file_list.pkl'), 'rb') as f:
                file_list = pickle.load(f)

        self.file_list = file_list

        self.image_dir = os.path.join(data_dir, 'image')
        self.segment_dir = os.path.join(data_dir, 'segment')

        self.joint_transform, self.input_transform, self.target_transform = self._get_transforms(dataset_stats)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        img_name = self.file_list[item]

        image = Image.open(join(self.image_dir, img_name))
        segmentation = Image.open(join(self.segment_dir, img_name))

        if self.joint_transform is not None:
            image, segmentation = self.joint_transform(image, segmentation)

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            segmentation = self.target_transform(segmentation)

        return image, segmentation

    def get_image_from_name(self, name: str):
        if name[-4:] != '.png':
            name = f'{name}.png'

        image = Image.open(join(self.image_dir, name))
        segmentation = Image.open(join(self.segment_dir, name))

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
                # custom_transforms.RandomHorizontalFlip(),
                # custom_transforms.Rotate()
            ])
        elif self.split == 'val':
            joint = None
        else:
            raise RuntimeError('Invalid dataset mode')

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(**mean_std)
        ])

        target_transform = standard_transforms.ToTensor()

        return joint, input_transform, target_transform


class MDSPrediction(Dataset):
    """Prediction Medical Scans Dataset"""

    def __init__(self, predict_path, dir_name, dataset_stats=None):
        dir_path = os.path.join(predict_path, dir_name)
        in_name = os.path.join(dir_path, f'{dir_name}_scan.npy')
        out_name = os.path.join(dir_path, f'{dir_name}_segmentation.npy')

        input_transforms, output_transforms = self._get_transforms(dataset_stats)

        self.scan = np.load(in_name)
        self.scan = np.transpose(self.scan, (1, 2, 0))
        self.scan = input_transforms(self.scan).type(torch.FloatTensor)
        self.scan = self.scan.unsqueeze(1)

        self.segmentation = np.load(out_name)
        self.segmentation = np.transpose(self.segmentation, (1, 2, 0))
        self.segmentation = output_transforms(self.segmentation).type(torch.FloatTensor)
        self.segmentation = self.segmentation.unsqueeze(1)

        self.shape = self.scan.shape

    def __len__(self) -> int:
        return len(self.scan)

    def __getitem__(self, item: int):
        image = self.scan[item, :, :]
        segment = self.segmentation[item, :, :]
        return image, segment

    @staticmethod
    def _get_transforms(mean_std):
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float().div(255)),
            transforms.Normalize(**mean_std)
        ])

        output_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float().div(255))
        ])

        return input_transform, output_transforms


class MDSDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, file_list=None, shuffle=True):
        assert config.mode in ['train', 'predict']

        self.config = config
        self.shuffle = shuffle
        self.input_channels = 1
        self.num_classes = 1
        self.image_size = (512, 512)

        data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(data_root, 'norm_data.pkl'), 'rb') as f:
            self.ds_statistics = pickle.load(f)

        if config.mode == 'train':
            self.train_path = get_dataset_path(config.path, config.dataset, 'train')
            val_path = get_dataset_path(config.path, config.dataset, 'val')

            self.train_dataset = MDSMain(self.train_path, split='train', file_list=file_list,
                                         dataset_stats=self.ds_statistics)
            self.val_dataset = MDSMain(val_path, split='val', dataset_stats=self.ds_statistics)

            self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                                           num_workers=config.num_workers,
                                           pin_memory=torch.cuda.is_available())

            self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size_val, shuffle=shuffle,
                                         num_workers=config.num_workers,
                                         pin_memory=torch.cuda.is_available())

            if file_list is None:
                self.msg = f'Train data loader created from {self.train_path}. ' \
                    f'Validation data loader created from {val_path}'
            else:
                self.msg = f'AL train data loader created from {self.train_path}. ' \
                    f'Validation data loader created from {val_path}'

            if config.run_val_on_train:
                self.val_train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size_val,
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
                                        self.ds_statistics)

        return DataLoader(predict_dataset,
                          batch_size=self.config.batch_size_val,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available())

    def update_train_loader(self, new_file_list: list) -> None:
        new_train_dataset = MDSMain(self.train_path, split='train', file_list=new_file_list,
                                    dataset_stats=self.ds_statistics)

        self.train_loader = DataLoader(new_train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.config.num_workers,
                                       pin_memory=torch.cuda.is_available())
