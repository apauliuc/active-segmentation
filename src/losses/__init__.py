from losses.BinaryCrossEntropyLoss2D import BinaryCrossEntropyLoss2D
from losses.SoftDiceLoss import SoftDiceLoss
import torch.nn as nn


def get_loss_fn(loss_dict):
    loss_fn_name = loss_dict['name']
    loss_fn = _get_model_instance(loss_fn_name)()

    return loss_fn


def _get_model_instance(name: str):
    try:
        return {
            'cross_entropy_loss': nn.CrossEntropyLoss,
            'bce_loss': nn.BCELoss,
            'bce_loss_2d': BinaryCrossEntropyLoss2D,
            'soft_dice_loss': SoftDiceLoss
        }[name]
    except KeyError:
        raise ('Loss function %s not available' % name)
