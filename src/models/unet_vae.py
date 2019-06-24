from models.unet_base import UNetBase
from models.common import FlattenLayer


class VariationalUNet(UNetBase):
    """
    Variational U-Net
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 image_size=(512, 512)):
        super(VariationalUNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                              False, 0)
        # U-Net modules constructed in the main method

        self.original_img_size = image_size
        # Create VAE part

    def forward(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)

        unet_out = self.unet_segmentation_map(self.unet_decoder(unet_encoding, previous_x))

        # TODO: do some more things here

        return unet_out

    def __repr__(self):
        return 'Variational U-Net'
