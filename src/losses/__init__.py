from losses.soft_dice import SoftDiceLoss
from losses.jaccard import JaccardLoss
from losses.bce_and_jaccard import BCEAndJaccardLoss
import torch.nn as nn


loss2class = {
    'bce_loss': nn.BCEWithLogitsLoss,
    'jaccard_loss': JaccardLoss,
    'bce_and_jaccard': BCEAndJaccardLoss,
    'cross_entropy_loss': nn.CrossEntropyLoss
    # 'bce_loss_2d': BinaryCrossEntropyLoss2D,
    # 'soft_dice_loss': SoftDiceLoss,
}


def get_loss_fn(loss_cfg):
    if loss_cfg.name is None:
        return nn.BCEWithLogitsLoss()
    else:
        loss_name = loss_cfg.name
        loss_params = {k: v for k, v in loss_cfg.items() if k != "name"}

        if loss_name not in loss2class:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        loss_fn = loss2class[loss_name](**loss_params)
        return loss_fn
