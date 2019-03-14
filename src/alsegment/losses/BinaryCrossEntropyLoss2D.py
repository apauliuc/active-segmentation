import torch.nn as nn


class BinaryCrossEntropyLoss2D(nn.Module):

    def __init__(self, reduction='mean'):
        super(BinaryCrossEntropyLoss2D, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, y_pred, y):
        batch_size = y_pred.size(0)
        y_pred = y_pred.view(batch_size, -1)
        y = y.view(batch_size, -1)
        return self.bce_loss(y_pred, y)

    def __repr__(self):
        return "BinaryCrossEntropy2D"
