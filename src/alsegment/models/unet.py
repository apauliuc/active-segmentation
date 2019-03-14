import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu2D(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super(ConvBnRelu2D, self).__init__()
        self._batch_norm = batch_norm
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self._batch_norm:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.relu(self.conv(x))


class UNetConvStack(nn.Module):
    """
    Convolution stack for UNet made of 2 ConvBnRelu2D modules
    """

    def __init__(self, in_size, out_size, batch_norm=False):
        super(UNetConvStack, self).__init__()

        self.conv_stack = nn.Sequential(
            ConvBnRelu2D(in_size, out_size, batch_norm),
            ConvBnRelu2D(out_size, out_size, batch_norm)
        )

    def forward(self, inputs):
        return self.conv_stack(inputs)


class UNetUpConvStack(nn.Module):
    """
    Up-convolution stack for UNet.
    Upsamples given input, appends past output, and applies UNetConvStack.
    """

    def __init__(self, in_size, out_size, use_conv=True):
        super(UNetUpConvStack, self).__init__()

        if use_conv:
            self.upsample = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = UNetConvStack(in_size, out_size)

    def forward(self, input1, input2):
        output1 = self.upsample(input1)

        diff_y = input2.size()[2] - output1.size()[2]
        diff_x = input2.size()[3] - output1.size()[3]

        output1 = F.pad(output1, [diff_x // 2, diff_x - diff_x // 2,
                                  diff_y // 2, diff_y - diff_y // 2])

        return self.conv(torch.cat((input2, output1), dim=1))


class UNet(nn.Module):
    """
    Implementation of the U-Net neural network for segmentation
    """

    def __init__(self, n_channels=1, n_classes=1, batch_norm=False, up_conv=True):
        super(UNet, self).__init__()

        # Maybe use batch norm on conv layers?

        filter_sizes = [64, 128, 256, 512, 1024]

        # Down sampling layers (1 to 4)
        self.conv1 = UNetConvStack(n_channels, filter_sizes[0], batch_norm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConvStack(filter_sizes[0], filter_sizes[1], batch_norm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConvStack(filter_sizes[1], filter_sizes[2], batch_norm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConvStack(filter_sizes[2], filter_sizes[3], batch_norm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottom convolution (5th layer)
        self.bottom_conv = UNetConvStack(filter_sizes[3], filter_sizes[4], batch_norm)

        # Up sampling layers (1 to 4) - concatenate past output
        self.upsample1 = UNetUpConvStack(filter_sizes[4], filter_sizes[3], up_conv)
        self.upsample2 = UNetUpConvStack(filter_sizes[3], filter_sizes[2], up_conv)
        self.upsample3 = UNetUpConvStack(filter_sizes[2], filter_sizes[1], up_conv)
        self.upsample4 = UNetUpConvStack(filter_sizes[1], filter_sizes[0], up_conv)

        # Final conv layer 1x1
        self.output_conv = nn.Conv2d(filter_sizes[0], n_classes, kernel_size=1)

        # Initialisation of weights with paper method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

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

        return self.output_conv(upsample4_out)

    def __repr__(self):
        return 'U-Net'
