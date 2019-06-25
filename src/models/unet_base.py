import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ConvBnRelu, get_upsampling_weight


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

        # Parameters
        filter_factors = [1, 2, 4, 8, 16]
        filter_sizes = [num_filters * s for s in filter_factors]
        self.filter_sizes = filter_sizes
        pool_layer = nn.MaxPool2d(2, 2)

        # Create downsampling layers
        self.down_path = nn.ModuleList()
        self.down_path.append(UNetConvBlock(self.in_channels,
                                            filter_sizes[0],
                                            batch_norm=batch_norm,
                                            dropout=False,
                                            dropout_p=dropout_p))
        self.down_samplers = [None]

        for prev_idx, num_filters in enumerate(filter_sizes[1:]):
            current_dropout = False if prev_idx == 0 else dropout
            t = 'encoder' if num_filters != filter_sizes[-1] else 'center'

            self.down_path.append(UNetConvBlock(filter_sizes[prev_idx],
                                                num_filters,
                                                batch_norm=batch_norm,
                                                dropout=current_dropout,
                                                dropout_p=dropout_p,
                                                block_type=t))
            self.down_samplers.append(pool_layer)

        # Create upsampling layers
        self.up_path = nn.ModuleList()
        self.up_samplers = nn.ModuleList()

        for idx, num_filters in enumerate(filter_sizes[1:]):
            current_dropout = False if idx < 2 else dropout

            self.up_path.append(UNetConvBlock(num_filters,
                                              filter_sizes[idx],
                                              batch_norm=batch_norm,
                                              dropout=current_dropout,
                                              dropout_p=dropout_p,
                                              block_type='decoder'))
            if learn_upconv:
                self.up_samplers.append(
                    nn.ConvTranspose2d(num_filters, filter_sizes[idx], kernel_size=4, stride=2, padding=1))
            else:
                self.up_samplers.append(
                    UpsampleConv(num_filters, filter_sizes[idx], scale_factor=2, mode='bilinear'))

            for m in self.unet_dec_up():
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.copy_(
                        get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                    )

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

    def forward(self, x):
        raise NotImplementedError
