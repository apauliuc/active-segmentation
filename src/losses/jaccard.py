import torch
import torch.nn as nn
import torch.nn.functional as F


class JaccardLoss(nn.Module):

    def __init__(self):
        super(JaccardLoss, self).__init__()

    # noinspection PyTypeChecker
    def forward(self, y_pred, y):
        num_classes = y_pred.shape[1]
        eps = 1e-7
        y = y.type(torch.LongTensor)

        if num_classes == 1:
            true_1_hot = torch.eye(2)[y.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat((true_1_hot_s, true_1_hot_f), dim=1)
            pos_proba = torch.sigmoid(y_pred)
            neg_proba = 1 - pos_proba
            probas = torch.cat((pos_proba, neg_proba), dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[y.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(y_pred, dim=1)

        true_1_hot = true_1_hot.type(y_pred.type())
        dims = (0,) + tuple(range(2, y.ndimension()))

        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection

        jacc_loss = (intersection / (union + eps)).mean()
        return 1 - jacc_loss

    def __repr__(self):
        return "Jaccard Loss"
