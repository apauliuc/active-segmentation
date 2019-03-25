import os
import pickle
import torch

from alsegment.data.MedicalScanDataset import MedicalScanDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from alsegment.data.data_transforms import ToPILImage
from alsegment.data.data_transforms import ToTensor
from alsegment.data.data_transforms import Normalize


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
                             num_workers=cfg['num_workers'],
                             pin_memory=torch.cuda.is_available())

    return data_loader
