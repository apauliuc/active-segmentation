from losses.soft_dice import SoftDiceLoss
from losses.jaccard import JaccardLoss
from losses.bce_and_jaccard import BCEAndJaccardLoss

import torch.nn as nn
from helpers.utils import retrieve_class_init_parameters


loss2class = {
    'bce_loss': nn.BCEWithLogitsLoss,
    'jaccard_loss': JaccardLoss,
    'bce_and_jaccard': BCEAndJaccardLoss,
    'cross_entropy_loss': nn.CrossEntropyLoss
    # 'bce_loss_2d': BinaryCrossEntropyLoss2D,
    # 'soft_dice_loss': SoftDiceLoss,
}


def get_loss_function(loss_cfg, gpu_node):
    loss_name = loss_cfg.name
    loss_cfg.gpu_node = gpu_node

    if loss_name is None:
        return nn.BCEWithLogitsLoss()
    else:
        if loss_name not in loss2class:
            raise NotImplementedError(f"Loss {loss_name} not implemented")

        loss_cls = loss2class[loss_name]

        init_param_names = retrieve_class_init_parameters(loss_cls)
        loss_params = {k: v for k, v in loss_cfg.items() if k in init_param_names}

        loss = loss_cls(**loss_params)

        return loss
