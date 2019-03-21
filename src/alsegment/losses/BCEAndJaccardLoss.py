from alsegment.losses.JaccardLoss import JaccardLoss
import torch.nn as nn


class BCEAndJaccardLoss(nn.Module):

    def __init__(self, weight=None):
        super(BCEAndJaccardLoss, self).__init__()
        self.jacc_loss_module = JaccardLoss()
        self.bce_loss_module = nn.BCEWithLogitsLoss()
        self.weight = weight

    def forward(self, y_pred, y):
        bce_loss = self.bce_loss_module(y_pred, y)
        jacc_loss = self.jacc_loss_module(y_pred, y)

        if self.weight is not None:
            total_loss = self.weight * bce_loss + (1 - self.weight) * jacc_loss
        else:
            total_loss = bce_loss + jacc_loss

        return total_loss
