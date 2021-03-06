import torch
import torch.nn.functional as F
from torch import nn as nn

from models.common import ConvBnRelu


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, dropout=False, dropout_p=0.2,
                 block_type='encoder'):
        super(UNetConvBlock, self).__init__()
        layers = []

        if dropout and block_type in ['encoder', 'center']:
            layers.append(nn.Dropout2d(p=dropout_p))

        layers.append(ConvBnRelu(in_size, out_size, batch_norm))
        layers.append(ConvBnRelu(out_size, out_size, batch_norm))

        if dropout and block_type in ['center', 'decoder']:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor=2, mode='bilinear'):
        super(UpsampleConv, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        x = self.conv(x)
        return x


class UNetBase(nn.Module):
    """
    Base class for U-Net neural networks

    Parameters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 dropout=False,
                 dropout_p=0.2):
        super(UNetBase, self).__init__()

        self.in_channels = input_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.mean, self.std = 0, 1

        # Parameters
        filter_factors = [1, 2, 4, 8, 16]
        filter_sizes = [num_filters * s for s in filter_factors]
        self.filter_sizes = filter_sizes
        pool_layer = nn.MaxPool2d(2, 2)

        # Create downsampling layers
        self.unet_enc_blocks = nn.ModuleList()
        self.unet_enc_blocks.append(UNetConvBlock(self.in_channels,
                                                  filter_sizes[0],
                                                  batch_norm=batch_norm,
                                                  dropout=False,
                                                  dropout_p=dropout_p))
        self.unet_enc_down = [None]

        for prev_idx, num_filters in enumerate(filter_sizes[1:]):
            current_dropout = False if prev_idx == 0 else dropout
            t = 'encoder' if num_filters != filter_sizes[-1] else 'center'

            self.unet_enc_blocks.append(UNetConvBlock(filter_sizes[prev_idx],
                                                      num_filters,
                                                      batch_norm=batch_norm,
                                                      dropout=current_dropout,
                                                      dropout_p=dropout_p,
                                                      block_type=t))
            self.unet_enc_down.append(pool_layer)

        # Create upsampling layers
        self.unet_dec_blocks = nn.ModuleList()
        self.unet_dec_up = nn.ModuleList()

        for idx, num_filters in enumerate(filter_sizes[1:]):
            current_dropout = False if idx < 2 else dropout

            self.unet_dec_blocks.append(UNetConvBlock(num_filters,
                                                      filter_sizes[idx],
                                                      batch_norm=batch_norm,
                                                      dropout=current_dropout,
                                                      dropout_p=dropout_p,
                                                      block_type='decoder'))
            if learn_upconv:
                self.unet_dec_up.append(
                    nn.ConvTranspose2d(num_filters, filter_sizes[idx], kernel_size=4, stride=2, padding=1))
            else:
                self.unet_dec_up.append(
                    UpsampleConv(num_filters, filter_sizes[idx], scale_factor=2, mode='bilinear'))

        # Final conv layer 1x1
        self.output_conv = nn.Conv2d(self.filter_sizes[0], self.num_classes, kernel_size=1)

    def unet_encoder(self, x):
        previous_x = []

        for enc_block, downsampler in zip(self.unet_enc_blocks, self.unet_enc_down):
            x_in = x if downsampler is None else downsampler(previous_x[-1])
            x_out = enc_block(x_in)
            previous_x.append(x_out)

        return previous_x[-1], previous_x

    def unet_decoder(self, x, previous_x):
        x_out = x
        for x_skip, dec_block, upsampler in reversed(list(zip(previous_x[:-1],
                                                              self.unet_dec_blocks,
                                                              self.unet_dec_up))):
            x_out = upsampler(x_out)
            x_out = dec_block(torch.cat([x_skip, x_out], dim=1))

        return x_out

    def unet_pipeline(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)
        unet_decoding = self.unet_decoder(unet_encoding, previous_x)
        return unet_encoding, unet_decoding

    def register_mean_std(self, mean_std, device):
        mean, std = mean_std.values()

        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)

        if self.in_channels > 1:
            self.mean = self.mean.unsqueeze(1).unsqueeze(1)
            self.std = self.std.unsqueeze(1).unsqueeze(1)

    def forward(self, x: torch.tensor, inference=False, num_samples=1):
        raise NotImplementedError
