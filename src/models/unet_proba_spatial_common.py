import torch
from torch import nn

from models.unet_base import UNetConvBlock
from models.unet_base_vae import VariationalUNetBase


# noinspection DuplicatedCode
class ProbaUNetSpatialCommon(VariationalUNetBase):
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
        super(ProbaUNetSpatialCommon, self).__init__(input_channels, num_classes, num_filters, batch_norm,
                                                     learn_upconv, latent_dim)
        upnet1_filters = [32, 16, 16, 8, 8]
        # upnet2_filters = [8, 4, 4]
        upnet2_filters = [8, 4]

        if upnet1_latent:
            upnet1_filters = [latent_dim for _ in range(5)]
            upnet2_filters[0] = latent_dim

        # Components for latent space
        if decrease_latent:
            latent_filters = [self.filter_sizes[-1], self.filter_sizes[-1] // 2]
        else:
            latent_filters = [self.filter_sizes[-1]] * 2

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
        self.create_f_combine(upnet1_filters[-1])

        # # # VAE Decoder

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

    def combine_features_and_sample(self, features, upnet1_output):
        maps_concat = torch.cat((features, upnet1_output), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def reconstruct_image(self, z):
        upnet1_output = self.upnet1(z)
        recon = self.upnet2(upnet1_output)
        return upnet1_output, recon.sub(self.mean).div(self.std)

    def _forward(self, x):
        unet_encoding, unet_decoding = self.unet_pipeline(x)

        mu, var = self.latent_space_params(unet_encoding)
        z = self.rsample(mu, var)

        upnet1_output, recon = self.reconstruct_image(z)

        combined_decoding = self.combine_features_and_sample(unet_decoding, upnet1_output)
        segmentation = self.output_conv(combined_decoding)

        return segmentation, recon, mu, var

    def _inference(self, x, num_samples=1):
        with torch.no_grad():
            unet_encoding, unet_decoding = self.unet_pipeline(x)

            mu, var = self.latent_space_params(unet_encoding)

            segmentation_all = []
            reconstruction_all = []

            for i in range(num_samples):
                z = self.rsample(mu, var)

                upnet1_output, recon = self.reconstruct_image(z)

                combined_decoding = self.combine_features_and_sample(unet_decoding, upnet1_output)

                segmentation_all.append(self.output_conv(combined_decoding))
                reconstruction_all.append(recon)

            return segmentation_all, reconstruction_all, mu, var

    def __repr__(self):
        return 'Probabilistic U-Net Spatial VAE Common Decoder'
