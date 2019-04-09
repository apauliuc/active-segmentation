from alsegment.losses.bce_2d import BinaryCrossEntropyLoss2D
from alsegment.losses.soft_dice import SoftDiceLoss
from alsegment.losses.jaccard import JaccardLoss
from alsegment.losses.bce_and_jaccard import BCEAndJaccardLoss
import torch.nn as nn


loss2class = {
    'bce_loss': nn.BCEWithLogitsLoss,
    'jaccard_loss': JaccardLoss,
    'bce_and_jaccard': BCEAndJaccardLoss,
    'cross_entropy_loss': nn.CrossEntropyLoss
    # 'bce_loss_2d': BinaryCrossEntropyLoss2D,
    # 'soft_dice_loss': SoftDiceLoss,
}


def get_loss_fn(loss_dict):
    if loss_dict['name'] is None:
        return nn.BCEWithLogitsLoss()
    else:
        loss_name = loss_dict['name']
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in loss2class:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        loss_fn = loss2class[loss_name](**loss_params)
        return loss_fn
