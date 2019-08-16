from typing import Tuple

import torch
from torch import nn

from models.common import ReparameterizedSample
from models.unet_base import UNetConvBlock
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

        upnet1_filters = [32, 16, 16, 8, 8]
        upnet2_filters = [8, 4, 4]

        if upnet1_latent:
            upnet1_filters = [16 for _ in range(5)]

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
        self.mean, self.std = 0, 1

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
            nn.Conv2d(upnet2_filters[-1], 1, kernel_size=1, stride=1))
        upnet2_layers.append(nn.Sigmoid())

        self.upnet2 = nn.Sequential(*upnet2_layers)

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

    def combine_features_and_sample(self, features, upnet1_output):
        """
        Features has shape (batch) x 32 x 512 x 512.
        z has size (batch)x(latent_dim)x(l_H)x(l_W).
        Use convTranspose2d layers to upsample to (batch) x 32 x 512 x 512
        Returns the result after applying conv layers on concatenated maps
        """
        maps_concat = torch.cat((features, upnet1_output), dim=self.channel_axis)
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

        upnet1_output = self.upnet1(z)

        # Combine sample with feature maps and get final segmentation map
        out_combined = self.combine_features_and_sample(unet_decoding, upnet1_output)
        segmentation = self.output_conv(out_combined)

        # Reconstruct image
        recon = self.upnet2(upnet1_output)

        return segmentation, recon, mu, var

    # def inference(self, x):
    #     with torch.no_grad():
    #         # Forward pass UNet encoder and decoder
    #         unet_encoding, previous_x = self.unet_encoder(x)
    #         unet_decoding = self.unet_decoder(unet_encoding, previous_x)
    #
    #         # Compute latent space parameters and sample
    #         mu, var = self.latent_sample(unet_encoding)
    #         z = self.rsample(mu, var)
    #
    #         # Create segmentation
    #         segmentation = self.output_conv(self.combine_features_and_sample(unet_decoding, z))
    #
    #         return segmentation, unet_decoding, mu, var
    #
    # def inference_multi(self, x, num_samples):
    #     with torch.no_grad():
    #         # Forward pass UNet encoder and decoder
    #         unet_encoding, previous_x = self.unet_encoder(x)
    #         unet_decoding = self.unet_decoder(unet_encoding, previous_x)
    #
    #         # Compute latent space parameters
    #         mu, var = self.latent_sample(unet_encoding)
    #
    #         # Create segmentations
    #         segmentations = []
    #         for i in range(num_samples):
    #             z = self.rsample(mu, var)
    #             current_segment = self.output_conv(self.combine_features_and_sample(unet_decoding, z))
    #             segmentations.append(current_segment)
    #
    #         return segmentations, unet_decoding, mu, var

    def __repr__(self):
        return 'Probabilistic U-Net Spatial Large'
