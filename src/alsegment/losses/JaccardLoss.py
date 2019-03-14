import torch
import torch.nn as nn


class JaccardLoss(nn.Module):

    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, y_pred, y):
        eps = 1e-15
        jacc_target = (y == 1).float()
        jacc_predicted = torch.sigmoid(y_pred)

        intersection = (jacc_predicted * jacc_target).sum()
        union = jacc_predicted.sum() + jacc_target.sum()

        loss = self.jacc_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss
