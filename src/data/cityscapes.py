import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import data.cityscapes_transforms as custom_transforms
import torchvision.transforms as standard_transforms

from data.base_loader import BaseLoader
from helpers.config import ConfigClass


def get_cityscapes_files(root, split):
    img_dir = os.path.join(root, 'leftImg8bit', split)
    segment_dir = os.path.join(root, 'gtFine', split)
    segment_postfix = '_gtFine_labelIds.png'

    items = []
    categories = os.listdir(segment_dir)

    for c in categories:
        if os.path.isdir(os.path.join(img_dir, c)):
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_dir, c))]

            for item in c_items:
                item = (os.path.join(img_dir, c, f'{item}_leftImg8bit.png'),
                        os.path.join(segment_dir, c, f'{item}{segment_postfix}'))

                items.append(item)
    return items, img_dir


class CityScapes(Dataset):
    classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
               "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    def __init__(self, root, split, file_list=None, input_size=(1024, 512)):
        self.split = split
        self.input_size = input_size
        self.ignore_label = 255

        if file_list is None:
            file_list, _ = get_cityscapes_files(root, split)

        self.images = file_list

        if len(self.images) == 0:
            raise RuntimeError('No images found. Check dataset.')

        self.joint_transform, self.input_transform, self.target_transform = self._get_transforms()

        self.id_to_trainId = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                              27: 14, 28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.images)

    # noinspection DuplicatedCode
    def __getitem__(self, index):
        image_path, segment_path = self.images[index]
        image, segmentation = Image.open(image_path).convert('RGB'), Image.open(segment_path)

        segmentation = np.array(segmentation)
        segmentation_copy = segmentation.copy()
        for k, v in self.id_to_trainId.items():
            segmentation_copy[segmentation == k] = v
        segmentation = Image.fromarray(segmentation_copy.astype(np.uint8))

        if self.joint_transform is not None:
            image, segmentation = self.joint_transform(image, segmentation)

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            segmentation = self.target_transform(segmentation)

        return image, segmentation

    def _get_transforms(self):
        mean_std = ([0.3006, 0.3365, 0.2956], [0.1951, 0.1972, 0.1968])

        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size),
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur()
            ])
        elif self.split == 'val':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        target_transform = custom_transforms.SegmentationToTensor()

        return joint, input_transform, target_transform


class CityScapesDataLoaders(BaseLoader):

    def __init__(self, config: ConfigClass, file_list=None, shuffle=True):

        self.config = config
        self.shuffle = shuffle
        self.input_channels = 3
        self.num_classes = 19
        self.image_size = (1024, 512)
        self.ds_statistics = {'mean': [0.3006, 0.3365, 0.2956], 'std': [0.1951, 0.1972, 0.1968]}

        # self.data_root = os.path.join(config.path, config.dataset)
        self.data_root = '/Users/andrei/Programming/CityscapesDataset/'

        self.train_dataset = CityScapes(self.data_root, 'train', file_list=file_list, input_size=self.image_size)
        self.val_dataset = CityScapes(self.data_root, 'val', input_size=self.image_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                                       num_workers=config.num_workers,
                                       pin_memory=torch.cuda.is_available())

        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size_val, shuffle=shuffle,
                                     num_workers=config.num_workers,
                                     pin_memory=torch.cuda.is_available())

        if file_list is None:
            self.msg = f'Data loaders created from {self.data_root}'
        else:
            self.msg = f'AL train data loader created from {self.data_root}'

        if config.run_val_on_train:
            self.val_train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size_val,
                                               shuffle=shuffle, num_workers=config.num_workers,
                                               pin_memory=torch.cuda.is_available())

    def update_train_loader(self, new_file_list: list) -> None:
        new_train_dataset = CityScapes(self.data_root, split='train', file_list=new_file_list)

        self.train_loader = DataLoader(new_train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.config.num_workers,
                                       pin_memory=torch.cuda.is_available())
