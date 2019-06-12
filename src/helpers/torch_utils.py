import torch
import torch.nn as nn

longTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def apply_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    x = x.to(device=device, non_blocking=non_blocking)
    y = y.to(device=device, non_blocking=non_blocking)
    return x, y
