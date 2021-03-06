from typing import Tuple

import torch
from torch import nn

VAE_CRITERION_FORWARD = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class KLDivergence(nn.Module):
    def __init__(self, prior_var=1):
        super(KLDivergence, self).__init__()
        self.prior_var = torch.as_tensor(prior_var, dtype=torch.float)

    def forward(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        device = mean.device
        num_el = mean.numel()

        if self.prior_var == 1:
            kld = torch.tensor(-0.5).to(device) * (1 + var.log() - mean.pow(2) - var).sum()
        else:
            kld = torch.tensor(-0.5).to(device) * (
                        1 + var.log() - self.prior_var.log() - (mean.pow(2) + var) / self.prior_var).sum()

        return kld / num_el


class VAECriterion(nn.Module):
    def __init__(self, ce_loss=nn.CrossEntropyLoss(reduction='mean'), mse_factor=0, kld_factor=0, prior_var=1):
        super(VAECriterion, self).__init__()

        self.mse_factor = mse_factor
        self.kld_factor = kld_factor

        self.ce = ce_loss
        self.mse = nn.MSELoss(reduction='mean')
        self.kld = KLDivergence(prior_var=prior_var)

    def forward(self, pred: torch.Tensor, y: torch.Tensor, recon: torch.Tensor, x: torch.Tensor,
                mu: torch.Tensor, var: torch.Tensor) -> VAE_CRITERION_FORWARD:
        ce = self.ce(pred, y)
        mse = self.mse(recon, x)
        kld = self.kld(mu, var)

        loss = ce + self.mse_factor * mse + self.kld_factor * kld

        return loss, ce, mse, kld

    def __repr__(self):
        return f'VAE Criterion with MSE factor {self.mse_factor} and KLD factor {self.kld_factor}'
