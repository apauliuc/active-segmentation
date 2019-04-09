import os
import pickle
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from alsegment.data.data_transforms import ToPILImage
from alsegment.data.data_transforms import ToTensor
from alsegment.data.data_transforms import Normalize


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


def create_data_loader(cfg, path, batch_size=4, shuffle=True, dataset=MedicalScanDataset):
    with open(os.path.join(os.path.dirname(path), 'norm_data.pkl'), 'rb') as f:
        norm = pickle.load(f)

    transf = transforms.Compose([
        ToPILImage(),
        ToTensor(),
        Normalize(norm['mean'], norm['std']),
    ])

    data_loader = DataLoader(dataset(path, transf),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=cfg.num_workers,
                             pin_memory=torch.cuda.is_available())

    return data_loader
