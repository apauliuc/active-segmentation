import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from models.unet_base import UNetBase


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
                                                False, 0)
        # U-Net components constructed in the main method

        # Components related to latent space
        self.latent_dim = latent_dim
        self.latent_space_conv = nn.Conv2d(self.filter_sizes[-1], 2*self.latent_dim, kernel_size=1, stride=1)

        # Components for combining sample with the feature maps
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        last_layers = [
            nn.Conv2d(self.filter_sizes[0] + self.latent_dim, self.filter_sizes[0], kernel_size=1),
            nn.ReLU(inplace=True)
        ]

        for _ in range(2):
            last_layers.append(nn.Conv2d(self.filter_sizes[0] + self.latent_dim, self.filter_sizes[0], kernel_size=1))
            last_layers.append(nn.ReLU(inplace=True))

        self.f_combine = nn.Sequential(*last_layers)

        # Components for input reconstruction
        self.recon_decoder = nn.Conv2d(self.latent_dim, self.filter_sizes[0], kernel_size=1)

    def construct_latent_distribution(self, encoding):
        # TODO combine information from previous layers (involves more parameters)

        # Shrink feature maps
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Compute mu and log sigma through conv layer
        mu_log_sigma = self.latent_space_conv(encoding)

        # Squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        latent_space_dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return latent_space_dist

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
        # TODO: finish this
        return self.recon_decoder(z)

    def forward(self, x, training=True):
        # Forward pass UNet encoder and decoder
        unet_encoding, previous_x = self.unet_encoder(x)
        unet_decoding = self.unet_decoder(unet_encoding, previous_x)

        # Compute latent space distribution and retrieve sample
        latent_space = self.construct_latent_distribution(unet_encoding)
        z_posterior = latent_space.rsample()

        # Combine sample with feature maps and get final segmentation map
        out_combined = self.combine_features_and_sample(unet_decoding, z_posterior)
        segmentation_map = self.output_conv(out_combined)

        # Reconstruct image
        reconstruction_input = self.reconstruction_decoder(z_posterior)

        return segmentation_map, latent_space, reconstruction_input

    def __repr__(self):
        return 'Probabilistic U-Net'
