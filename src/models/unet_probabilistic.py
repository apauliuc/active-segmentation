from typing import Tuple

import numpy as np
import torch
from torch import nn

from models.common import ReparameterizedSample
from models.unet_base import UNetBase

PUNET_FORWARD = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
                 latent_dim=6):
        super(ProbabilisticUNet, self).__init__(input_channels, num_classes, num_filters, batch_norm, learn_upconv,
                                                dropout=False, dropout_p=0)
        # U-Net components constructed in the main method

        # Components for latent space
        self.latent_dim = latent_dim
        self.latent_space_conv = nn.Conv2d(self.filter_sizes[-1], 2 * self.latent_dim, kernel_size=1, stride=1)
        self.var_softplus = nn.Softplus()
        self.rsample = ReparameterizedSample()

        # Components for combining sample with feature maps
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        last_layers = [
            nn.Conv2d(self.filter_sizes[0] + self.latent_dim, self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter_sizes[0], self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True)
        ]

        self.f_combine = nn.Sequential(*last_layers)

        # Components for input reconstruction
        self.mean, self.std = 0, 1
        decoder_filters = [256, 128, 64, 32, 16, 8, 4, 2]
        decoder_layers = [
            nn.ConvTranspose2d(self.latent_dim, decoder_filters[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]

        for idx, filter_size in enumerate(decoder_filters[1:]):
            decoder_layers.append(
                nn.ConvTranspose2d(decoder_filters[idx], filter_size, kernel_size=4, stride=2, padding=1))

            decoder_layers.append(nn.ReLU(inplace=True))

        decoder_layers.append(
            nn.ConvTranspose2d(decoder_filters[-1], 1, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.recon_decoder = nn.Sequential(*decoder_layers)

    def latent_sample(self, encoding):
        # Can combine information from previous layers (involves more parameters)

        # encoding shape: batch x 512 x 32 x 32

        # Shrink feature maps
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        # shape: batch x 512 x 1 x 1

        # AVG POOL
        # b x 512 x 32 x 32 -> avg pool (2x2) + conv (kernel 3, padding 1, stride 1)
        # -> b x 512 x 16 x 16 -> avg pool (2x2) + conv (kernel 3, padding 1, stride 1)
        # -> b x 512 x 8 x 8 -> conv (kernel 1, stride 1, padding 0)
        # -> b x 2 * latent_size x 8 x 8

        # mean/std shape: b x latent_channels x latent_H x latent_W

        # shape: batch x latent_size x 8 x 8

        # Compute mu and log sigma through conv layer
        mu_var = self.latent_space_conv(encoding)
        # shape: batch x 2*latent size x 1 x 1

        # Squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_var = torch.squeeze(mu_var, dim=2)
        mu_var = torch.squeeze(mu_var, dim=2)

        mu = mu_var[:, :self.latent_dim]
        var = self.var_softplus(mu_var[:, self.latent_dim:]) + 1e-5  # lower bound variance of posterior

        return mu, var

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        # noinspection PyArgumentList
        order_index = torch.LongTensor(order_index).to(next(self.parameters()).device)
        return torch.index_select(a, dim, order_index)

    def combine_features_and_sample(self, features, z):
        """
        Features has shape (batch)x(channels)x(H)x(W), z has shape (batch)x(latent_dim).
        Broadcast Z to (batch)x(latent_dim)x(H)x(W) using behaviour as in tf.tile
        Returns the result after applying conv layers on concatenated maps
        """
        z = torch.unsqueeze(z, self.spatial_axes[0])
        z = self.tile(z, self.spatial_axes[0], features.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, self.spatial_axes[1])
        z = self.tile(z, self.spatial_axes[1], features.shape[self.spatial_axes[1]])

        maps_concat = torch.cat((features, z), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def reconstruction_decoder(self, z):
        z = torch.unsqueeze(z, self.spatial_axes[0])
        z = torch.unsqueeze(z, self.spatial_axes[1])

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
        recon = self.reconstruction_decoder(z)
        # z shape: b x latent_channels x 1 x 1

        return segmentation, recon, mu, var

    def __repr__(self):
        return 'Probabilistic U-Net'
