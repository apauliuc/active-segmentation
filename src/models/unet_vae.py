import torch
from torch import nn

from models.unet_base import UNetBase
from models.common import ConvBnRelu, ReparameterizedSample, UNetConvBlock


# noinspection DuplicatedCode
class VariationalUNet(UNetBase):
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
                                              False, 0)
        # U-Net modules constructed in the main method

        self.latent_dim = latent_dim
        self.channel_axis = 1
        self.mean, self.std = 0, 1

        if not bottom_block_full:
            self.unet_enc_blocks[-1] = ConvBnRelu(self.filter_sizes[-2], self.filter_sizes[-1], batch_norm)

        self.latent_space_pipeline = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.filter_sizes[-1], 2 * self.latent_dim, kernel_size=3, stride=1, padding=1)
        )

        self.var_softplus = nn.Softplus()
        self.rsample = ReparameterizedSample()

        self.upconv_sample = nn.ConvTranspose2d(self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1)

        if not bottom_block_full:
            self.combine_encoding_latent = ConvBnRelu(self.filter_sizes[-1] + self.latent_dim, self.filter_sizes[-1],
                                                      batch_norm)
        else:
            self.combine_encoding_latent = UNetConvBlock(self.filter_sizes[-1] + self.latent_dim, self.filter_sizes[-1],
                                                         batch_norm)

        self.recon_pipeline = nn.Sequential(
            nn.Conv2d(self.filter_sizes[0], input_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def latent_sample(self, encoding):
        # encoding shape: batch x 512 x 32 x 32

        mu_var = self.latent_space_pipeline(encoding)

        mu = mu_var[:, :self.latent_dim, :, :]
        var = self.var_softplus(mu_var[:, self.latent_dim:, :, :]) + 1e-5  # lower bound variance of posterior

        return mu, var

    def combine_features_and_sample(self, features, z):
        z = self.upconv_sample(z)
        maps_concat = torch.cat((features, z), dim=self.channel_axis)
        return self.combine_encoding_latent(maps_concat)

    def reconstruct_image(self, dec):
        recon = self.recon_pipeline(dec)
        return recon.sub(self.mean).div(self.std)

    def forward(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)

        mu, var = self.latent_sample(unet_encoding)
        z = self.rsample(mu, var)

        combined_encoding = self.combine_features_and_sample(unet_encoding, z)

        decoding = self.unet_decoder(combined_encoding, previous_x)

        segmentation = self.output_conv(decoding)
        recon = self.reconstruct_image(decoding)

        return segmentation, recon, mu, var

    def inference(self, x):
        with torch.no_grad():
            unet_encoding, previous_x = self.unet_encoder(x)

            mu, var = self.latent_sample(unet_encoding)
            z = self.rsample(mu, var)

            combined_encoding = self.combine_features_and_sample(unet_encoding, z)
            decoding = self.unet_decoder(combined_encoding, previous_x)

            segmentation = self.output_conv(decoding)

            return segmentation, decoding, mu, var

    def inference_multi(self, x, num_samples):
        with torch.no_grad():
            unet_encoding, previous_x = self.unet_encoder(x)

            mu, var = self.latent_sample(unet_encoding)

            segmentations = []
            decodings = []
            for i in range(num_samples):
                z = self.rsample(mu, var)
                decoding = self.unet_decoder(self.combine_features_and_sample(unet_encoding, z), previous_x)
                current_segment = self.output_conv(decoding)

                segmentations.append(current_segment)
                decodings.append(decoding)

            return segmentations, decodings, mu, var

    def __repr__(self):
        return 'Variational U-Net'
