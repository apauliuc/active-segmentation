from alsegment.mds_patient_pool import ALMDSPatientPool
from alsegment.cityscapes_pool import ALCityScapesPool


def get_pool_class(data_cfg):
    if 'AMC' in data_cfg.dataset:
        return ALMDSPatientPool(data_cfg)

    elif 'CityScapes' in data_cfg.dataset:
        return ALCityScapesPool(data_cfg)
