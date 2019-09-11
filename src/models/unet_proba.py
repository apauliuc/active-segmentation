import numpy as np
import torch
from torch import nn

from models.unet_base_vae import VariationalUNetBase


class ProbabilisticUNet(VariationalUNetBase):
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
                                                latent_dim)
        # Components for latent space
        self.latent_space_conv = nn.Conv2d(self.filter_sizes[-1], 2 * self.latent_dim, kernel_size=1, stride=1)

        # Components for combining sample with feature maps
        self.create_f_combine(self.latent_dim)

        # Components for input reconstruction
        self.create_recon_pipeline([256, 128, 64, 64, 32, 32, 16, 16, 8])

    def latent_space_pipeline(self, encoding):
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_var = self.latent_space_conv(encoding)

        mu_var = torch.squeeze(mu_var, dim=2)
        mu_var = torch.squeeze(mu_var, dim=2)

        return mu_var

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
        for ax in self.spatial_axes:
            z = torch.unsqueeze(z, ax)
            z = self.tile(z, ax, features.shape[ax])

        maps_concat = torch.cat((features, z), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def reconstruct_image(self, z):
        z = torch.unsqueeze(z, self.spatial_axes[0])
        z = torch.unsqueeze(z, self.spatial_axes[1])

        return super().reconstruct_image(z)

    def __repr__(self):
        return 'Probabilistic U-Net'
