import torch

longTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
floatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# floatTensor = torch.FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
