from typing import Tuple

import torch
from torch import nn

VAE_CRITERION_FORWARD = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        device = mean.device
        num_el = mean.numel()

        kld = torch.tensor(-0.5).to(device) * (1 + var.log() - mean.pow(2) - var).sum()

        return kld / num_el


class VAECriterion(nn.Module):
    def __init__(self, ce_loss=nn.CrossEntropyLoss(reduction='mean')):
        super(VAECriterion, self).__init__()

        self.ce = ce_loss
        self.mse = nn.MSELoss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, pred: torch.Tensor, y: torch.Tensor, recon: torch.Tensor, x: torch.Tensor,
                mu: torch.Tensor, var: torch.Tensor) -> VAE_CRITERION_FORWARD:
        ce = self.ce(pred, y)
        mse = self.mse(recon, x)
        kld = self.kld(mu, var)

        return ce, mse, kld
