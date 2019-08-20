from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

from models.common import ReparameterizedSample
from models.unet_base import UNetBase, UNetConvBlock


# noinspection DuplicatedCode
class VariationalUNetBase(UNetBase):
    """
    Base class for U-Net neural networks with Variational Component
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 latent_dim=6):
        super(VariationalUNetBase, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                                  False, 0)
        self.latent_dim = latent_dim
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        self.mean, self.std = 0, 1

        self.var_softplus = nn.Softplus()
        self.rsample = ReparameterizedSample()

        self.latent_space_pipeline = nn.Sequential()
        self.upconv_sample = nn.Sequential()
        self.recon_pipeline = nn.Sequential()
        self.f_combine = nn.Sequential()

    def register_mean_std(self, mean_std, device):
        mean, std = mean_std.values()

        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)

    def latent_space_params(self, encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_var = self.latent_space_pipeline(encoding)

        mu = mu_var[:, :self.latent_dim]
        var = self.var_softplus(mu_var[:, self.latent_dim:]) + 1e-5  # lower bound variance of posterior

        return mu, var

    def combine_features_and_sample(self, features, z):
        z = self.upconv_sample(z)
        maps_concat = torch.cat((features, z), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def reconstruct_image(self, z):
        recon = self.recon_pipeline(z)
        return recon.sub(self.mean).div(self.std)

    def forward(self, x: torch.tensor, inference=False, num_samples=1):
        if inference:
            return self._inference(x, num_samples)
        else:
            return self._forward(x)

    def _forward(self, x):
        unet_encoding, unet_decoding = self.unet_pipeline(x)

        mu, var = self.latent_space_params(unet_encoding)
        z = self.rsample(mu, var)

        combined_decoding = self.combine_features_and_sample(unet_decoding, z)

        segmentation = self.output_conv(combined_decoding)
        recon = self.reconstruct_image(z)

        return segmentation, recon, mu, var

    def _inference(self, x, num_samples=1):
        with torch.no_grad():
            unet_encoding, unet_decoding = self.unet_pipeline(x)

            mu, var = self.latent_space_params(unet_encoding)

            segmentation_all = []
            reconstruction_all = []

            for i in range(num_samples):
                z = self.rsample(mu, var)

                combined_decoding = self.combine_features_and_sample(unet_decoding, z)

                segmentation_all.append(self.output_conv(combined_decoding))
                reconstruction_all.append(self.reconstruct_image(z))

            return segmentation_all, reconstruction_all, mu, var

    def create_f_combine(self, new_ch_number):
        self.f_combine = nn.Sequential(
            nn.Conv2d(self.filter_sizes[0] + new_ch_number, self.filter_sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter_sizes[0], self.filter_sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def create_upconv_sample(self, no_upconv_sample):
        upconv_sample = []

        for _ in range(no_upconv_sample):
            upconv_sample.append(
                nn.ConvTranspose2d(self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1))
            upconv_sample.append(nn.ReLU(inplace=True))
            upconv_sample.append(UNetConvBlock(self.latent_dim, self.latent_dim))

        self.upconv_sample = nn.Sequential(*upconv_sample)

    def create_recon_pipeline(self, decoder_filters, last_kernel_size=1):
        decoder_layers = [
            nn.ConvTranspose2d(self.latent_dim, decoder_filters[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            UNetConvBlock(decoder_filters[0], decoder_filters[0])
        ]

        for idx, filter_size in enumerate(decoder_filters[1:]):
            decoder_layers.append(
                nn.ConvTranspose2d(decoder_filters[idx], filter_size, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(UNetConvBlock(filter_size, filter_size))

        if last_kernel_size == 1:
            decoder_layers.append(
                nn.Conv2d(decoder_filters[-1], self.in_channels, kernel_size=1, stride=1))
        else:
            decoder_layers.append(
                nn.Conv2d(decoder_filters[-1], self.in_channels, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.recon_pipeline = nn.Sequential(*decoder_layers)
