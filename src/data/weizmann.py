import os
import pickle
import numpy as np
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from data.base_loader import BaseLoader
from helpers.config import ConfigClass
import data.cityscapes_transforms as custom_transforms
import torchvision.transforms as standard_transforms


class WeizmannDS(Dataset):
    """Weizmann Segmentation Dataset"""

    def __init__(self, data_dir, split, file_list=None, dataset_stats=None, input_size=(288, 112)):
        self.split = split
        self.input_size = input_size
        self.root_dir = join(data_dir, split)

        self.segmentation_files = []
        self.seg_to_img = {}
        self.img_to_seg = {}

        if file_list is None:
            for x in os.listdir(self.root_dir):
                if os.path.isdir(join(self.root_dir, x)):
                    segments = os.listdir(join(self.root_dir, x, 'human_seg'))
                    self.segmentation_files.extend(segments)
                    self.img_to_seg[x] = segments

                    for s in segments:
                        self.seg_to_img[s] = x

        self.joint_transform, self.input_transform, self.target_transform = self._get_transforms(dataset_stats)

    def __len__(self) -> int:
        return len(self.segmentation_files)

    def __getitem__(self, item: int):
        segmentation_name = self.segmentation_files[item]
        image_name = self.seg_to_img[segmentation_name]

        image = Image.open(join(self.root_dir, image_name, 'src_color', f'{image_name}.png'))
        segmentation = Image.open(join(self.root_dir, image_name, 'human_seg', segmentation_name))

        if self.joint_transform is not None:
            image, segmentation = self.joint_transform(image, segmentation)

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            segmentation = np.where(np.array(segmentation)[:, :, 1] != 0, 0, 1).astype(np.float32)
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
            # standard_transforms.Resize((self.input_size[1], self.input_size[0])),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(**mean_std)
        ])

        target_transform = standard_transforms.Compose([
            # standard_transforms.Resize((self.input_size[1], self.input_size[0])),
            standard_transforms.ToTensor()
        ])

        return joint, input_transform, target_transform

    def get_image_from_name(self, name: str):
        segmentation_names = self.img_to_seg[name]
        pairs = []

        for seg_name in segmentation_names:
            image = Image.open(join(self.root_dir, name, 'src_color', f'{name}.png'))
            segmentation = Image.open(join(self.root_dir, name, 'human_seg', seg_name))

            if self.joint_transform is not None:
                image, segmentation = self.joint_transform(image, segmentation)

            if self.input_transform is not None:
                image = self.input_transform(image)

            if self.target_transform is not None:
                segmentation = np.where(np.array(segmentation)[:, :, 1] != 0, 0, 1).astype(np.float32)
                segmentation = self.target_transform(segmentation)

            pairs.append((image, segmentation))

        return pairs


class WeizmannDSEvaluation(Dataset):
    """Weizmann Segmentation Dataset - Evaluation phase"""

    def __init__(self, data_dir, dataset_stats=None, input_size=(288, 112)):
        self.input_size = input_size
        self.root_dir = data_dir

        self.image_list = []

        for x in os.listdir(self.root_dir):
            if os.path.isdir(join(self.root_dir, x)):
                self.image_list.append(x)

        self.joint_transform, self.input_transform, self.target_transform = self._get_transforms(dataset_stats)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, item: int):
        image_name = self.image_list[item]

        image = Image.open(join(self.root_dir, image_name, 'src_color', f'{image_name}.png'))
        shape = (image.height, image.width)

        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, image_name, shape

    def _get_transforms(self, mean_std):
        input_transform = standard_transforms.Compose([
            standard_transforms.Resize((self.input_size[1], self.input_size[0])),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(**mean_std)
        ])

        return None, input_transform, None


class WeizmannDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, file_list=None, shuffle=True):

        self.config = config
        self.shuffle = shuffle
        self.input_channels = 3
        self.num_classes = 1
        self.image_size = (288, 112)  # width x height

        self.data_root = os.path.join(config.path, config.dataset)

        with open(os.path.join(self.data_root, 'norm_data.pkl'), 'rb') as f:
            ds_statistics = pickle.load(f)

        self.train_dataset = WeizmannDS(self.data_root, 'train', file_list=file_list,
                                        dataset_stats=ds_statistics, input_size=self.image_size)
        self.val_dataset = WeizmannDS(self.data_root, 'val',
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

    def update_train_loader(self, new_file_list: list) -> None:
        new_train_dataset = WeizmannDS(self.data_root, split='train', file_list=new_file_list,
                                       dataset_stats=self.ds_statistics)

        self.train_loader = DataLoader(new_train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.config.num_workers,
                                       pin_memory=torch.cuda.is_available())
