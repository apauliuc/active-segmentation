import torch
from torch import nn

from models.unet_base import UNetConvBlock
from models.common import ConvBnRelu
from models.unet_base_vae import VariationalUNetBase


class VariationalUNet(VariationalUNetBase):
    """
    Variational U-Net
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 latent_dim=6,
                 bottom_block_full=True):
        super(VariationalUNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                              latent_dim)
        # Components for latent space
        if not bottom_block_full:
            self.unet_enc_blocks[-1] = ConvBnRelu(self.filter_sizes[-2], self.filter_sizes[-1], batch_norm)

        self.latent_space_pipeline = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.filter_sizes[-1], 2 * self.latent_dim, kernel_size=3, stride=1, padding=1)
        )

        self.upconv_sample = nn.ConvTranspose2d(self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1)

        if not bottom_block_full:
            self.f_combine = ConvBnRelu(self.filter_sizes[-1] + self.latent_dim, self.filter_sizes[-1],
                                        batch_norm)
        else:
            self.f_combine = UNetConvBlock(self.filter_sizes[-1] + self.latent_dim, self.filter_sizes[-1],
                                           batch_norm)

        # Components for input reconstruction
        self.recon_pipeline = nn.Sequential(
            nn.Conv2d(self.filter_sizes[0], input_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def _forward(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)

        mu, var = self.latent_space_params(unet_encoding)
        z = self.rsample(mu, var)

        combined_encoding = self.combine_features_and_sample(unet_encoding, z)

        decoding = self.unet_decoder(combined_encoding, previous_x)

        segmentation = self.output_conv(decoding)
        recon = self.reconstruct_image(decoding)

        return segmentation, recon, mu, var

    def _inference(self, x: torch.tensor, num_samples=1):
        with torch.no_grad():
            unet_encoding, previous_x = self.unet_encoder(x)

            mu, var = self.latent_space_params(unet_encoding)

            segmentation_all = []
            reconstruction_all = []

            for i in range(num_samples):
                z = self.rsample(mu, var)

                combined_encoding = self.combine_features_and_sample(unet_encoding, z)
                decoding = self.unet_decoder(combined_encoding, previous_x)

                segmentation_all.append(self.output_conv(decoding))
                reconstruction_all.append(self.reconstruct_image(decoding))

            return segmentation_all, reconstruction_all, mu, var

    def __repr__(self):
        return 'Variational U-Net'
