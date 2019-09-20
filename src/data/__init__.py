from data.cityscapes import CityScapesDataLoaders
from data.medical_scans import MDSDataLoaders
from data.msra10k import MSRA10KDataLoaders
from data.weizmann import WeizmannDataLoaders
from data.voc import VOCDataLoader


def get_dataloaders(data_cfg, file_list=None, shuffle=True):
    if 'AMC' in data_cfg.dataset:
        return MDSDataLoaders(data_cfg, file_list=file_list, shuffle=shuffle)

    elif 'CityScapes' in data_cfg.dataset:
        return CityScapesDataLoaders(data_cfg, file_list=file_list, shuffle=shuffle)

    elif 'Weizmann' in data_cfg.dataset:
        return WeizmannDataLoaders(data_cfg, file_list=file_list, shuffle=shuffle)

    elif 'MSRA10K' in data_cfg.dataset:
        return MSRA10KDataLoaders(data_cfg, file_list=file_list, shuffle=shuffle)
