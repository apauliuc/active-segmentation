import copy

from models.unet import UNet
from models.ffnn import FeedFwdNeuralNet


def get_model(model_dict, n_channels=1, n_classes=1):
    model_name = model_dict['name']
    model = _get_model_instance(model_name)
    param_dict = model_dict['network_params']

    if model_name == 'unet':
        model = model(n_channels=n_channels, n_classes=n_classes, **param_dict)

    # elif model_name == 'ffnn':
    #     model = model()

    return model


def _get_model_instance(name: str):
    try:
        return {
            'unet': UNet,
            'ffnn': FeedFwdNeuralNet
        }[name]
    except KeyError:
        raise ('Model %s not available' % name)
