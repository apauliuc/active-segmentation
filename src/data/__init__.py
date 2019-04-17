from data.medical_dataset import MDSDataLoaders
from data.voc import VOCDataLoader


def get_dataloaders(data_cfg, file_list=None, shuffle=True):
    if 'AMC' in data_cfg.dataset:
        return MDSDataLoaders(data_cfg, file_list=file_list, shuffle=shuffle)

    # elif 'VOC' in data_cfg.dataset:
    #     return VOCDataLoader(data_cfg)
