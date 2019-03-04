import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_augmentation import ToPILImage
from data_augmentation import FlipNumpy
from data_augmentation import Flip
from data_augmentation import ToTensor
from definitions import DATA_DIR


class MedicalScanDataset(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, root_dir=DATA_DIR, split='train', transform=lambda x: x):
        """

        :param root_dir: path to folder containing input images
        :param split: specify whether to take train/test dataset
        :param transform: optional transform to apply on samples
        """
        assert split in ['train', 'test']

        self.data_dir = join(root_dir, split)

        with open(join(root_dir, 'partition.pkl'), 'rb') as f:
            partition = pickle.load(f)

        self.file_list = partition[split]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        img_name = self.file_list[item]

        image = np.load(join(self.data_dir, img_name + '.npy'))
        segmentation = np.load(join(self.data_dir, img_name + '_seg.npy'))

        image, segmentation = self.transform((image, segmentation))

        return {'image': image, 'segmentation': segmentation, 'name': img_name}


if __name__ == '__main__':

    train_dataset = MedicalScanDataset(DATA_DIR, split='train')

    for i in range(len(train_dataset)):
        scan, seg = train_dataset[i]['image'], train_dataset[i]['segmentation']

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(scan)

        plt.subplot(1, 2, 2)
        plt.imshow(seg)

        plt.show()

        break
