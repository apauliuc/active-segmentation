from models.unet_base import UNetBase
from models.common import FlattenLayer


class ProbabilisticUNet(UNetBase):
    """
    Probabilistic U-Net
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 image_size=(512, 512)):
        super(ProbabilisticUNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                                False, 0)
        # U-Net modules constructed in the main method

        self.original_img_size = image_size
        # Create VAE part

    def forward(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)

        unet_decoding = self.unet_decoder(unet_encoding, previous_x)

        # TODO: do some more things here

        unet_out = self.output_conv(unet_decoding)

        return unet_out

    def __repr__(self):
        return 'Variational U-Net'
