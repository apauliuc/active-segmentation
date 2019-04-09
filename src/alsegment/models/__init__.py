from alsegment.models.unet import UNet
from alsegment.models.ternausnet import TernausNet
from alsegment.models.unet_v2 import UNetV2
from alsegment.models.segnet import SegNet


def get_model(model_cfg, n_channels=1, n_classes=1):
    model_name = model_cfg.name
    model = _get_model_instance(model_name)
    param_dict = {k: v for k, v in model_cfg.network_params.items()}

    if model_name == 'unet':
        model = model(n_channels=n_channels, n_classes=n_classes, **param_dict)

    elif model_name in ['ternaus_net', 'unet_v2', 'segnet']:
        model = model(**param_dict)

    return model


def _get_model_instance(name: str):
    try:
        return {
            'unet': UNet,
            'ternaus_net': TernausNet,
            'unet_v2': UNetV2,
            'segnet': SegNet
        }[name]
    except KeyError:
        raise Exception(f'Model {name} not available')
