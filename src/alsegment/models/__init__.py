from alsegment.models.unet import UNet
from alsegment.models.ffnn import FeedFwdNeuralNet
from alsegment.models.ternausnet import TernausNet
from alsegment.models.unet_v2 import UNetV2


def get_model(model_dict, n_channels=1, n_classes=1):
    model_name = model_dict['name']
    model = _get_model_instance(model_name)
    param_dict = model_dict['network_params'] if model_dict['network_params'] is not None else {}

    if model_name == 'unet':
        model = model(n_channels=n_channels, n_classes=n_classes, **param_dict)

    elif model_name == 'ternaus_net':
        model = model(**param_dict)

    elif model_name == 'unet_v2':
        model = model(**param_dict)

    # elif model_name == 'ffnn':
    #     model = model()

    return model


def _get_model_instance(name: str):
    try:
        return {
            'unet': UNet,
            'ternaus_net': TernausNet,
            'unet_v2': UNetV2,
            'ffnn': FeedFwdNeuralNet
        }[name]
    except KeyError:
        raise ('Model %s not available' % name)
