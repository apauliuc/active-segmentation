import numpy as np
import torch
from torch import distributions, nn
from typing import Tuple


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBnRelu(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


def initialize_weights(*m):
    for model in m:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, _in: torch.Tensor):
        bs = _in.shape[0]
        return _in.reshape((bs, -1))


class Unflatten(nn.Module):
    shape: Tuple[int, ...]

    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, _in: torch.Tensor, num_samples: int, shape: Tuple[int, ...]) -> torch.Tensor:
        bs = _in.shape[0]

        return _in.reshape((bs, max(1, num_samples), *shape))


class ReparameterizedSample(nn.Module):
    def __init__(self):
        super(ReparameterizedSample, self).__init__()

    def forward(self, mean: torch.Tensor, var: torch.Tensor, num_samples=1) -> torch.Tensor:
        std = torch.sqrt(var)
        dist = distributions.normal.Normal(mean, std)
        if num_samples == 1:
            z = dist.rsample()
        else:
            z = dist.rsample([num_samples])
            z = z.transpose(0, 1)

        return z.contiguous()


class DropoutLayer(nn.Dropout2d):

    def __init__(self, dropout_full, dropout_p):
        super(DropoutLayer, self).__init__(p=dropout_p, inplace=False)
        self.dropout_full = dropout_full
        self.dropout_p = dropout_p

    def forward(self, x):
        if self.dropout_full:
            if self.training:
                mask = torch.zeros_like(x) if np.random.rand(1) < self.dropout_p else torch.ones_like(x)
                return x * mask
            else:
                return x * (1. - self.dropout_p)
        else:
            return super(DropoutLayer, self).forward(x)
