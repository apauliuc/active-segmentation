import torch.nn as nn
from models.components import UNetConv2d, UNetUpConcatConv2d


class UNet(nn.Module):
    """
    Implementation of the U-Net neural network for segmentation
    """

    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()

        # Maybe use batch norm on conv layers?

        filter_sizes = [64, 128, 256, 512, 1024]

        # Down sampling layers (1 to 4)
        self.conv1 = UNetConv2d(n_channels, filter_sizes[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2d(filter_sizes[0], filter_sizes[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2d(filter_sizes[1], filter_sizes[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2d(filter_sizes[2], filter_sizes[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottom convolution (5th layer)
        self.bottom_conv = UNetConv2d(filter_sizes[3], filter_sizes[4])

        # Up sampling layers (1 to 4) - concatenate past output
        self.upsample1 = UNetUpConcatConv2d(filter_sizes[4], filter_sizes[3], False)
        self.upsample2 = UNetUpConcatConv2d(filter_sizes[3], filter_sizes[2], False)
        self.upsample3 = UNetUpConcatConv2d(filter_sizes[2], filter_sizes[1], False)
        self.upsample4 = UNetUpConcatConv2d(filter_sizes[1], filter_sizes[0], False)

        # Final conv layer 1x1
        self.last_conv = nn.Conv2d(filter_sizes[0], n_classes, kernel_size=1)

    def forward(self, in_image):
        conv1_out = self.conv1(in_image)
        conv2_out = self.conv2(self.maxpool1(conv1_out))
        conv3_out = self.conv3(self.maxpool2(conv2_out))
        conv4_out = self.conv4(self.maxpool3(conv3_out))

        bottom = self.bottom_conv(self.maxpool4(conv4_out))

        upsample1_out = self.upsample1(bottom, conv4_out)
        upsample2_out = self.upsample2(upsample1_out, conv3_out)
        upsample3_out = self.upsample3(upsample2_out, conv2_out)
        upsample4_out = self.upsample4(upsample3_out, conv1_out)

        return self.last_conv(upsample4_out)

    def __repr__(self):
        return 'U-Net'
