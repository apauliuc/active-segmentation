import torch.nn as nn


class FeedFwdNeuralNet(nn.Module):

    def __init__(self):
        super(FeedFwdNeuralNet, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(262144, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 262144),
            nn.Sigmoid()
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x.reshape(orig_shape)

    def __repr__(self):
        return 'Feed Forward Neural Network'
