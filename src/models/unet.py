from models.unet_base import UNetBase


class UNet(UNetBase):
    """
    Standard U-Net model for semantic segmentation
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 dropout=False,
                 dropout_p=0.2):
        super(UNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                   dropout, dropout_p)

    def forward(self, x, inference=False, num_samples=1):
        unet_encoding, previous_x = self.unet_encoder(x)

        unet_out = self.output_conv(self.unet_decoder(unet_encoding, previous_x))

        return unet_out

    def __repr__(self):
        return 'U-Net with Dropout' if self.dropout else 'Standard U-Net'
