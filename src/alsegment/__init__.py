from alsegment.mds_patient_pool import ALMDSPatientPool
from alsegment.cityscapes_pool import ALCityScapesPool
from alsegment.msra10k_pool import ALMSRA10KPool


def get_pool_class(config):
    if 'AMC' in config.data.dataset:
        return ALMDSPatientPool(config)

    elif 'CityScapes' in config.data.dataset:
        return ALCityScapesPool(config)

    elif 'MSRA10K' in config.data.dataset:
        return ALMSRA10KPool(config)
