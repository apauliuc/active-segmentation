from typing import Tuple

import numpy as np
import torch
from torch import nn

from models.common import ReparameterizedSample
from models.unet_base import UNetBase

PUNET_FORWARD = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# noinspection DuplicatedCode
class ProbabilisticUNetBigger(UNetBase):
    """
    Probabilistic U-Net
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 latent_dim=6):
        super(ProbabilisticUNetBigger, self).__init__(input_channels, num_classes, num_filters, batch_norm,
                                                      learn_upconv,
                                                      dropout=False, dropout_p=0)
        # U-Net components constructed in the main method

        # Components for latent space
        self.latent_dim = latent_dim

        self.latent_space_pipeline = nn.Sequential(
            nn.Conv2d(self.filter_sizes[-1], self.filter_sizes[-1] // 2, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.Conv2d(self.filter_sizes[-1] // 2, self.filter_sizes[-1] // 4, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2),
            nn.Conv2d(self.filter_sizes[-1] // 4, 2 * self.latent_dim, kernel_size=1, stride=1)
        )

        self.var_softplus = nn.Softplus()
        self.rsample = ReparameterizedSample()

        # Components for combining sample with feature maps
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        sample_upscale = []

        for _ in range(6):
            sample_upscale.append(
                nn.ConvTranspose2d(self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1))
            sample_upscale.append(nn.ReLU(inplace=True))

        self.sample_upscale = nn.Sequential(*sample_upscale)

        self.f_combine = nn.Sequential(
            nn.Conv2d(self.filter_sizes[0] + self.latent_dim, self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter_sizes[0], self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Components for input reconstruction
        self.mean, self.std = 0, 1
        decoder_filters = [32, 32, 16, 8, 4, 2]
        decoder_layers = [
            nn.ConvTranspose2d(self.latent_dim, decoder_filters[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]

        for idx, filter_size in enumerate(decoder_filters[1:]):
            decoder_layers.append(
                nn.ConvTranspose2d(decoder_filters[idx], filter_size, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU(inplace=True))

        decoder_layers.append(
            nn.Conv2d(decoder_filters[-1], 1, kernel_size=1, stride=1))
        decoder_layers.append(nn.Sigmoid())

        self.recon_decoder = nn.Sequential(*decoder_layers)

    def latent_sample(self, encoding):
        # Can combine information from previous layers (involves more parameters)

        # encoding shape: batch x 512 x 32 x 32

        # # # Spatial information # # #

        # using AVG POOL
        # b x 512 x 32 x 32 -> conv (kernel 3, padding 1, stride 1) + avg pool (2x2)
        # -> b x 512 x 16 x 16 -> conv (kernel 3, padding 1, stride 1) + avg pool (2x2)
        # -> b x 512 x 8 x 8 -> conv (kernel 1, stride 1, padding 0)
        # -> b x 2 * latent_size x 8 x 8

        mu_var = self.latent_space_pipeline(encoding)

        mu = mu_var[:, :self.latent_dim, :, :]
        var = self.var_softplus(mu_var[:, self.latent_dim:, :, :]) + 1e-5  # lower bound variance of posterior

        return mu, var

    def combine_features_and_sample(self, features, z):
        """
        Features has shape (batch)x(channels)x(H)x(W).
        z has size (batch)x(latent_dim)x(l_H)x(l_W).
        Use convTranspose2d layers to upsample to (batch)x(latent_dim)x(H)x(W)
        Returns the result after applying conv layers on concatenated maps
        """
        z = self.sample_upscale(z)
        maps_concat = torch.cat((features, z), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def reconstruct_image(self, z):
        recon = self.recon_decoder(z)
        return recon.sub(self.mean).div(self.std)

    def forward(self, x) -> PUNET_FORWARD:
        # Forward pass UNet encoder and decoder
        unet_encoding, previous_x = self.unet_encoder(x)
        unet_decoding = self.unet_decoder(unet_encoding, previous_x)

        # Compute latent space parameters and sample
        mu, var = self.latent_sample(unet_encoding)
        z = self.rsample(mu, var)

        # Combine sample with feature maps and get final segmentation map
        out_combined = self.combine_features_and_sample(unet_decoding, z)
        segmentation = self.output_conv(out_combined)

        # Reconstruct image
        recon = self.reconstruct_image(z)

        return segmentation, recon, mu, var

    def __repr__(self):
        return 'Probabilistic U-Net BIG'
