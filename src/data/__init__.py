from data.medical_dataset import MDSDataLoaders
from data.voc import VOCDataLoader


def get_dataloaders(data_cfg, shuffle=True):
    if 'AMC' in data_cfg.dataset:
        return MDSDataLoaders(data_cfg, shuffle)

    elif 'VOC' in data_cfg.dataset:
        return VOCDataLoader(data_cfg)
