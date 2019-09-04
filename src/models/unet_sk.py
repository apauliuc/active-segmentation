import torch

from models.common import DropoutLayer
from models.unet_base import UNetBase


# noinspection DuplicatedCode
class SKUNet(UNetBase):
    """
    Standard U-Net model with Stochastic Skip Connections for semantic segmentation
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 dropout=False,
                 dropout_p=0.2,
                 dropout_full=False):
        super(SKUNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                     dropout, dropout_p)

        self.dropout_layers = []

        for i in range(len(self.unet_enc_down)):
            self.dropout_layers.append(DropoutLayer(dropout_full=dropout_full, dropout_p=dropout_p))

        self.last_dropout = DropoutLayer(dropout_full=dropout_full, dropout_p=dropout_p)

    def unet_decoder(self, x, previous_x):
        x_out = x
        for x_skip, dec_block, upsampler, dropout_l in reversed(list(zip(previous_x[:-1],
                                                                         self.unet_dec_blocks,
                                                                         self.unet_dec_up,
                                                                         self.dropout_layers))):
            x_out = upsampler(x_out)
            x_out = dec_block(torch.cat([dropout_l(x_skip), x_out], dim=1))

        return x_out

    def forward(self, x, inference=False, num_samples=1):
        unet_encoding, previous_x = self.unet_encoder(x)

        unet_out = self.output_conv(self.unet_decoder(unet_encoding, previous_x))

        return unet_out

    def __repr__(self):
        return 'SKUNet'
