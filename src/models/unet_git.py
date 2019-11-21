import torch
from torch import nn


class UNetOtherVersion(nn.Module):
    """UNet implementation from GitHub"""

    @staticmethod
    def contracting_block(in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    @staticmethod
    def expansive_block(in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )
        return block

    @staticmethod
    def final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def __init__(self, input_channels, num_classes):
        super(UNetOtherVersion, self).__init__()

        self.conv_encode1 = UNetOtherVersion.contracting_block(in_channels=input_channels, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = UNetOtherVersion.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = UNetOtherVersion.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = UNetOtherVersion.expansive_block(512, 256, 128)
        self.conv_decode2 = UNetOtherVersion.expansive_block(256, 128, 64)
        self.final_layer = UNetOtherVersion.final_block(128, 64, num_classes)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)

        return final_layer

    @staticmethod
    def concat(up, bypass):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        return torch.cat((up, bypass), 1)

    def __repr__(self):
        return 'UNet Git'
