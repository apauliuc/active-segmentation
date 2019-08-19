from typing import Tuple

import torch
from torch import nn

from models.common import ReparameterizedSample, UNetConvBlock
from models.unet_base import UNetBase

PUNET_FORWARD = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
PUNET_INFERENCE = Tuple[torch.Tensor]


# noinspection DuplicatedCode
class ProbaUNetSpCommon(UNetBase):
    """
    Probabilistic U-Net
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 latent_dim=6,
                 decrease_latent=False,
                 upnet1_latent=False):
        super(ProbaUNetSpCommon, self).__init__(input_channels, num_classes, num_filters, batch_norm,
                                                learn_upconv,
                                                dropout=False, dropout_p=0)
        # U-Net components constructed in the main method

        self.mean, self.std = 0, 1
        upnet1_filters = [32, 16, 16, 8, 8]
        upnet2_filters = [8, 4, 4]

        if upnet1_latent:
            upnet1_filters = [latent_dim for _ in range(5)]
            upnet2_filters[0] = latent_dim

        # Components for latent space
        self.latent_dim = latent_dim

        if decrease_latent:
            self.latent_space_pipeline = nn.Sequential(
                nn.Conv2d(self.filter_sizes[-1], self.filter_sizes[-1] // 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(2),
                nn.Conv2d(self.filter_sizes[-1] // 2, 2 * self.latent_dim, kernel_size=1, stride=1)
            )
        else:
            self.latent_space_pipeline = nn.Sequential(
                nn.Conv2d(self.filter_sizes[-1], self.filter_sizes[-1], kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(2),
                nn.Conv2d(self.filter_sizes[-1], 2 * self.latent_dim, kernel_size=1, stride=1)
            )

        self.var_softplus = nn.Softplus()
        self.rsample = ReparameterizedSample()

        # Components for combining sample with feature maps
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        self.f_combine = nn.Sequential(
            nn.Conv2d(self.filter_sizes[0] + upnet1_filters[-1], self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter_sizes[0], self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # VAE Decoder

        # UpNet1
        upnet1_layers = [
            nn.ConvTranspose2d(self.latent_dim, upnet1_filters[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            UNetConvBlock(upnet1_filters[0], upnet1_filters[0])
        ]

        for idx, filter_size in enumerate(upnet1_filters[1:]):
            upnet1_layers.append(
                nn.ConvTranspose2d(upnet1_filters[idx], filter_size, kernel_size=4, stride=2, padding=1))
            upnet1_layers.append(nn.ReLU(inplace=True))
            upnet1_layers.append(UNetConvBlock(filter_size, filter_size))

        self.upnet1 = nn.Sequential(*upnet1_layers)

        # UpNet2
        upnet2_layers = []

        for idx, filter_size in enumerate(upnet2_filters[1:]):
            upnet2_layers.append(
                nn.Conv2d(upnet2_filters[idx], filter_size, kernel_size=3, stride=1, padding=1))
            upnet2_layers.append(nn.ReLU(inplace=True))

        upnet2_layers.append(
            nn.Conv2d(upnet2_filters[-1], input_channels, kernel_size=1, stride=1))
        upnet2_layers.append(nn.Sigmoid())

        self.upnet2 = nn.Sequential(*upnet2_layers)

    def latent_sample(self, encoding):
        mu_var = self.latent_space_pipeline(encoding)

        mu = mu_var[:, :self.latent_dim, :, :]
        var = self.var_softplus(mu_var[:, self.latent_dim:, :, :]) + 1e-5  # lower bound variance of posterior

        return mu, var

    def combine_features_and_sample(self, features, upnet1_output):
        maps_concat = torch.cat((features, upnet1_output), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def vae_decoder(self, z):
        upnet1_output = self.upnet1(z)
        recon = self.upnet2(upnet1_output)
        return upnet1_output, recon.sub(self.mean).div(self.std)

    def forward(self, x) -> PUNET_FORWARD:
        unet_encoding, unet_decoding = self.unet_pipeline(x)

        mu, var = self.latent_sample(unet_encoding)
        z = self.rsample(mu, var)

        upnet1_output, recon = self.vae_decoder(z)

        out_combined = self.combine_features_and_sample(unet_decoding, upnet1_output)
        segmentation = self.output_conv(out_combined)

        return segmentation, recon, mu, var

    def inference(self, x):
        with torch.no_grad():
            unet_encoding, unet_decoding = self.unet_pipeline(x)

            mu, var = self.latent_sample(unet_encoding)
            z = self.rsample(mu, var)

            upnet1_output = self.upnet1(z)

            segmentation = self.output_conv(self.combine_features_and_sample(unet_decoding, upnet1_output))

            return segmentation, unet_decoding, mu, var

    def inference_multi(self, x, num_samples):
        with torch.no_grad():
            unet_encoding, unet_decoding = self.unet_pipeline(x)

            mu, var = self.latent_sample(unet_encoding)

            # Create segmentations
            segmentations = []
            for i in range(num_samples):
                z = self.rsample(mu, var)
                upnet1_output = self.upnet1(z)
                current_segment = self.output_conv(self.combine_features_and_sample(unet_decoding, upnet1_output))
                segmentations.append(current_segment)

            return segmentations, unet_decoding, mu, var

    def __repr__(self):
        return 'Probabilistic U-Net Spatial Common Decoder'
