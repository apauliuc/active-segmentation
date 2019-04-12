import os

import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms

import data.voc_utils as extended_transforms
from data.base_loader import BaseLoader
from data.voc_utils import make_dataset
from definitions import VOC_ROOT


class VOC(data.Dataset):
    def __init__(self, mode, data_root=VOC_ROOT, joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None):
        self.imgs = make_dataset(mode, data_root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)


class VOCDataLoader(BaseLoader):

    def __init__(self, config):
        assert config.mode in ['train', 'test', 'random']

        self.input_channels = 3
        self.num_classes = 21

        data_root = os.path.join(config.path, config.dataset)

        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        def tmp_mul(x):
            return x.mul_(255)

        self.input_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            extended_transforms.FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(tmp_mul),
            standard_transforms.Normalize(*mean_std)
        ])

        self.target_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
            extended_transforms.MaskToTensor()
        ])

        self.restore_transform = standard_transforms.Compose([
            extended_transforms.DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            extended_transforms.FlipChannels()
        ])

        self.visualize = standard_transforms.Compose([
            standard_transforms.Resize(400),
            standard_transforms.CenterCrop(400),
            standard_transforms.ToTensor()
        ])
        if config.mode == 'random':
            train_data = torch.randn(config.batch_size, config.input_channels, config.img_size,
                                     config.img_size)
            train_labels = torch.ones(config.batch_size, config.img_size, config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + config.batch_size - 1) // config.batch_size
            self.valid_iterations = (self.len_valid_data + config.batch_size - 1) // config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.val_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif config.mode == 'train':
            train_dataset = VOC('train', data_root,
                                transform=self.input_transform, target_transform=self.target_transform)
            val_dataset = VOC('val', data_root,
                              transform=self.input_transform, target_transform=self.target_transform)

            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                           num_workers=config.num_workers,
                                           pin_memory=torch.cuda.is_available())
            self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False,
                                         num_workers=config.num_workers,
                                         pin_memory=torch.cuda.is_available())
            self.train_iterations = (len(train_dataset) + config.batch_size) // config.batch_size
            self.val_iterations = (len(val_dataset) + config.batch_size) // config.batch_size

            self.msg = f'Train data loader created from {data_root}'

            if config.run_val_on_train:
                self.val_train_loader = DataLoader(train_dataset, batch_size=config.batch_size_val,
                                                   shuffle=True, num_workers=config.num_workers,
                                                   pin_memory=torch.cuda.is_available())

        elif config.mode == 'test':
            test_set = VOC('test', config.data_root,
                           transform=self.input_transform, target_transform=self.target_transform)

            self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                          num_workers=config.num_workers,
                                          pin_memory=torch.cuda.is_available())
            self.test_iterations = (len(test_set) + config.batch_size) // config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')
