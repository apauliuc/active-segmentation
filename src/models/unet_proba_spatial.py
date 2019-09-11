import torch
from torch import nn

from models.unet_base_vae import VariationalUNetBase


class ProbabilisticUNetSpatial(VariationalUNetBase):
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
                 spatial_large=False,
                 decrease_latent=False,
                 last_kernel_size=3):
        super(ProbabilisticUNetSpatial, self).__init__(input_channels, num_classes, num_filters, batch_norm,
                                                       learn_upconv, latent_dim)

        if spatial_large:
            no_upconv_sample = 5
            decoder_filters = [32, 16, 8, 4, 2]

            if decrease_latent:
                latent_filters = [self.filter_sizes[-1], self.filter_sizes[-1] // 2]
            else:
                latent_filters = [self.filter_sizes[-1]] * 2
        else:
            no_upconv_sample = 6
            decoder_filters = [32, 32, 16, 8, 4, 2]

            if decrease_latent:
                latent_filters = [self.filter_sizes[-1], self.filter_sizes[-1] // 2, self.filter_sizes[-1] // 4]
            else:
                latent_filters = [self.filter_sizes[-1]] * 3

        # Components for latent space
        temp_pipeline = []

        for idx, k in enumerate(latent_filters):
            if idx != len(latent_filters) - 1:
                temp_pipeline.append(
                    nn.Conv2d(k, latent_filters[idx + 1], kernel_size=3, stride=1, padding=1))
                temp_pipeline.append(nn.AvgPool2d(2))
            else:
                temp_pipeline.append(
                    nn.Conv2d(k, 2 * self.latent_dim, kernel_size=1, stride=1))

        self.latent_space_pipeline = nn.Sequential(*temp_pipeline)

        # Components for combining sample with feature maps
        self.create_upconv_sample(no_upconv_sample)

        # Components for combining sample with feature maps
        self.create_f_combine(self.latent_dim)

        # Components for input reconstruction
        self.create_recon_pipeline(decoder_filters, last_kernel_size)

    def __repr__(self):
        return 'Probabilistic U-Net Spatial'
