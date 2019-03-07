import pickle
import numpy as np
from os.path import join
from torch.utils.data import Dataset


class MedicalScanDataset(Dataset):
    """Medical Scans Dataset"""

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: path to folder containing images
        :param transform: optional transform to apply on samples
        """
        self.n_channels = 1
        self.n_classes = 2

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
