from torch import sigmoid
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y):
        smooth = 1.
        batch_size = y_pred.size(0)
        y_pred = sigmoid(y_pred).view(batch_size, -1)
        y = y.view(batch_size, -1)
        intersection = (y_pred * y).sum()

        score = (2. * intersection.sum(1) + smooth) / (y_pred.sum(1) + y.sum(1) + smooth)
        score = 1 - score.sum() / batch_size
        return score

    def __repr__(self):
        return "SoftDiceLoss"
