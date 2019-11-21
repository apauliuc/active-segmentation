import math
import torch
import torch.nn as nn
from torch.nn import init

from models.common import DropoutLayer

longTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def apply_dropout(m):
    if type(m) == nn.Dropout2d or type(m) == DropoutLayer:
        m.train()


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    x = x.to(device=device, non_blocking=non_blocking)
    y = y.to(device=device, non_blocking=non_blocking)
    return x, y


# noinspection PyProtectedMember
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            m.uniform_(m.bias, -bound, bound)
