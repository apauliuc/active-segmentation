import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian.layers import BBBConv2d


class BBBUNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(BBBUNetConvBlock, self).__init__()
        layers = [
            BBBConv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            BBBConv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl = 0

        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
            else:
                x = layer(x)

        return x, kl


class BBBUpsampleConv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor=2, mode='bilinear'):
        super(BBBUpsampleConv, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = BBBConv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        x, kl = self.conv.convprobforward(x)
        return x, kl


class BBBUnet(nn.Module):

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 learn_upconv=False):
        super(BBBUnet, self).__init__()
        self.in_channels = input_channels
        self.num_classes = num_classes

        # Parameters
        filter_factors = (1, 2, 4, 8, 16)
        filter_sizes = [num_filters * s for s in filter_factors]
        pool_layer = nn.MaxPool2d(2, 2)

        # Create downsampling layers
        self.down_path = nn.ModuleList()
        self.down_samplers = [None]
        self.down_path.append(BBBUNetConvBlock(self.in_channels,
                                               filter_sizes[0]))

        for prev_idx, num_filters in enumerate(filter_sizes[1:]):
            self.down_samplers.append(pool_layer)
            self.down_path.append(BBBUNetConvBlock(filter_sizes[prev_idx],
                                                   num_filters))

        # Create upsampling layers
        self.up_path = nn.ModuleList()
        self.up_samplers = nn.ModuleList()

        for idx, num_filters in enumerate(filter_sizes[1:]):
            self.up_path.append(BBBUNetConvBlock(num_filters,
                                                 filter_sizes[idx]))
            self.up_samplers.append(
                BBBUpsampleConv(num_filters, filter_sizes[idx], scale_factor=2, mode='bilinear'))

        # Final conv layer 1x1
        self.output_conv = BBBConv2d(filter_sizes[0], self.num_classes, kernel_size=1)

    def forward(self, x):
        logits, kl = self.probforward(x)
        return logits

    def probforward(self, x):
        previous_x = []
        kl = 0

        for downsample, down in zip(self.down_samplers, self.down_path):
            x_in = x if downsample is None else downsample(previous_x[-1])
            x_out, _kl = down(x_in)
            kl += _kl
            previous_x.append(x_out)

        x_out = previous_x[-1]
        for x_skip, upsample, up in reversed(list(zip(previous_x[:-1], self.up_samplers, self.up_path))):
            x_out, _kl_up = upsample(x_out)
            x_out, _kl = up(torch.cat([x_skip, x_out], dim=1))
            kl += (_kl + _kl_up)

        x_out, _kl = self.output_conv.convprobforward(x_out)
        kl += _kl

        logits = x_out
        return logits, kl

    def __repr__(self):
        return 'Bayesian UNet'
