import torch
import torch.nn as nn

longTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()
