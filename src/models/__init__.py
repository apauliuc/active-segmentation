from models.unet import UNet
from models.ternausnet import TernausNet
from models.unet_v2 import UNetV2
from models.segnet import SegNet
from models.fcn8 import FCN8
from models.fcn16 import FCN16
from models.fcn32 import FCN32
from bayesian.bayes_unet import BBBUnet

from helpers.utils import retrieve_class_init_parameters


def get_model(model_cfg):
    model_name = model_cfg.name
    model_cls = _get_model_instance(model_name)

    init_param_names = retrieve_class_init_parameters(model_cls)
    param_dict = {k: v for k, v in model_cfg.network_params.items() if k in init_param_names}

    model = model_cls(**param_dict)

    return model


def _get_model_instance(name: str):
    try:
        return {
            'unet': UNet,
            'ternaus_net': TernausNet,
            'unet_v2': UNetV2,
            'segnet': SegNet,
            'fcn8': FCN8,
            'fcn16': FCN16,
            'fcn32': FCN32,
            'bbb_unet': BBBUnet
        }[name]
    except KeyError:
        raise Exception(f'Model {name} not available')
