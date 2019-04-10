from alsegment.data.medical_dataset import MDSDataLoaders
from alsegment.data.voc import VOCDataLoader


def get_dataloaders(data_cfg, shuffle=True):
    if 'AMC' in data_cfg.dataset:
        return MDSDataLoaders(data_cfg, shuffle)

    elif 'VOC' in data_cfg.dataset:
        return VOCDataLoader(data_cfg)
