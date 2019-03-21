import numpy as np
from torch.utils.data import Dataset

import os
import pickle
import torch

from torch.utils.data import DataLoader
from torchvision import transforms


class MedicalScanPrediction(Dataset):
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


def create_prediction_loader(data_path, file_name, shuffle=False, batch_size=4):
    with open(os.path.join(os.path.dirname(data_path), 'norm_data.pkl'), 'rb') as f:
        norm = pickle.load(f)

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([norm['mean']], [norm['std']]),
    ])

    data_loader = DataLoader(MedicalScanPrediction(os.path.join(data_path, file_name), transf),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=2,
                             pin_memory=torch.cuda.is_available())

    return data_loader
