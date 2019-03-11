import os
import pickle

from dataset import MedicalScanDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from data_transforms import ToPILImage
from data_transforms import FlipNumpy
from data_transforms import Flip
from data_transforms import ToTensor
from data_transforms import Normalize


def create_data_loader(cfg, path, shuffle=True, dataset=MedicalScanDataset):
    with open(os.path.join(os.path.dirname(path), 'norm_data.pkl'), 'rb') as f:
        norm = pickle.load(f)

    transf = transforms.Compose([
        ToPILImage(),
        ToTensor(),
        Normalize(norm['mean'], norm['std']),
    ])

    data_loader = DataLoader(dataset(path, transf),
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['num_workers'],
                             shuffle=shuffle)

    return data_loader
