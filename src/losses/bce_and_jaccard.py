import torch
import torch.nn as nn
from losses import JaccardLoss


class BCEAndJaccardLoss(nn.Module):

    def __init__(self, weight=None, eval_ensemble=False, gpu_node=0):
        super(BCEAndJaccardLoss, self).__init__()
        device = torch.device(f'cuda:{gpu_node}' if torch.cuda.is_available() else 'cpu')
        self.jacc_loss_module = JaccardLoss(eval_ensemble=eval_ensemble, device=device).to(device=device)
        if eval_ensemble:
            self.bce_loss_module = nn.BCELoss().to(device=device)
        else:
            self.bce_loss_module = nn.BCEWithLogitsLoss().to(device=device)
        self.weight = weight

    def forward(self, y_pred, y):
        bce_loss = self.bce_loss_module(y_pred, y)
        jacc_loss = self.jacc_loss_module(y_pred, y)

        if self.weight is not None:
            total_loss = self.weight * bce_loss + (1 - self.weight) * jacc_loss
        else:
            total_loss = bce_loss + jacc_loss

        return total_loss

    def __repr__(self):
        return "BCE and Jaccard Loss"
