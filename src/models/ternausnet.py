import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Parameters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, dropout=False, dropout_p=0.2):
        super(DecoderBlock, self).__init__()
        self.is_deconv = is_deconv
        module_list = [ConvRelu(in_channels, middle_channels)]

        if is_deconv:
            module_list.append(nn.ConvTranspose2d(middle_channels, out_channels,
                                                  kernel_size=4, stride=2, padding=1))
            module_list.append(nn.ReLU(inplace=True))
        else:
            module_list.append(ConvRelu(middle_channels, out_channels))

        if dropout:
            module_list.append(nn.Dropout2d(p=dropout_p))

        self.block = nn.Sequential(*module_list)

    def forward(self, x):
        if not self.is_deconv:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return self.block(x)


class TernausNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=False,
                 dropout=True,
                 dropout_p=0.2):
        """
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11

        Implementation from https://github.com/ternaus/robot-surgery-segmentation/blob/master/models.py
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.relu,
        )
        self.conv2 = nn.Sequential(
            self.encoder[3],
            self.relu
        )
        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu
        )
        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu
        )

        if dropout:
            layers_drop = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
            for l in layers_drop:
                l.add_module('dropout', nn.Dropout2d(p=self.dropout_p))

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8,
                                   dropout=dropout, dropout_p=dropout_p)
        self.dec5 = DecoderBlock(num_filters * 8 * 3, num_filters * 8 * 2, num_filters * 8,
                                 dropout=dropout, dropout_p=dropout_p)
        self.dec4 = DecoderBlock(num_filters * 8 * 3, num_filters * 8 * 2, num_filters * 4,
                                 dropout=dropout, dropout_p=dropout_p)
        self.dec3 = DecoderBlock(num_filters * 4 * 3, num_filters * 4 * 2, num_filters * 2,
                                 dropout=dropout, dropout_p=dropout_p)
        self.dec2 = DecoderBlock(num_filters * 2 * 3, num_filters * 2 * 2, num_filters,
                                 dropout=dropout, dropout_p=dropout_p)
        self.dec1 = ConvRelu(num_filters * 3, num_filters)

        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat((center, conv5), 1))
        dec4 = self.dec4(torch.cat((dec5, conv4), 1))
        dec3 = self.dec3(torch.cat((dec4, conv3), 1))
        dec2 = self.dec2(torch.cat((dec3, conv2), 1))
        dec1 = self.dec1(torch.cat((dec2, conv1), 1))

        return self.final(dec1)

    def __repr__(self):
        return 'TernausNet'
