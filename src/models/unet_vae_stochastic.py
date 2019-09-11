import torch

from models import VariationalUNetNoRecon
from models.common import DropoutLayer


class StochasticUNetNoRecon(VariationalUNetNoRecon):
    """
    Variational U-Net without reconstruction with Stochastic Skip Connections
    """

    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=32,
                 batch_norm=False,
                 learn_upconv=True,
                 latent_dim=6,
                 bottom_block_full=True,
                 dropout_p=0.25,
                 dropout_full=True):
        super(StochasticUNetNoRecon, self).__init__(input_channels, num_classes, num_filters, batch_norm,
                                                    learn_upconv, latent_dim, bottom_block_full)
        self.dropout_layers = []

        for i in range(len(self.unet_enc_down)):
            self.dropout_layers.append(DropoutLayer(dropout_full=dropout_full, dropout_p=dropout_p))

        self.last_dropout = DropoutLayer(dropout_full=dropout_full, dropout_p=dropout_p)

    def unet_decoder(self, x, previous_x):
        x_out = x
        for x_skip, dec_block, upsampler, dropout_l in reversed(list(zip(previous_x[:-1],
                                                                         self.unet_dec_blocks,
                                                                         self.unet_dec_up,
                                                                         self.dropout_layers))):
            x_out = upsampler(x_out)
            x_out = dec_block(torch.cat([dropout_l(x_skip), x_out], dim=1))

        return x_out

    def combine_features_and_sample(self, features, z):
        z = self.upconv_sample(z)
        maps_concat = torch.cat((self.last_dropout(features), z), dim=self.channel_axis)
        return self.f_combine(maps_concat)

    def _forward(self, x):
        unet_encoding, previous_x = self.unet_encoder(x)

        mu, var = self.latent_space_params(unet_encoding)
        z = self.rsample(mu, var)

        combined_encoding = self.combine_features_and_sample(unet_encoding, z)
        decoding = self.unet_decoder(combined_encoding, previous_x)

        segmentation = self.output_conv(decoding)

        return segmentation, None, mu, var

    def _inference(self, x: torch.tensor, num_samples=1):
        with torch.no_grad():
            unet_encoding, previous_x = self.unet_encoder(x)

            mu, var = self.latent_space_params(unet_encoding)

            segmentation_all = []

            for i in range(num_samples):
                z = self.rsample(mu, var)

                combined_encoding = self.combine_features_and_sample(unet_encoding, z)
                decoding = self.unet_decoder(combined_encoding, previous_x)

                segmentation_all.append(self.output_conv(decoding))

            return segmentation_all, [], mu, var

    def __repr__(self):
        return 'Stochastic sKip connection UNet - No Reconstruction'
