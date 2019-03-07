import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConv2d(nn.Module):
    """
    Convolution layer for UNet made of 2 Conv2d layers
    """

    def __init__(self, in_size, out_size):
        super(UNetConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)


class UNetUpConcatConv2d(nn.Module):
    """
    Up-convolution layer for UNet.
    Upsamples given input, appends past output, and applies UNetConv2d.
    """

    def __init__(self, in_size, out_size, use_conv=True):
        super(UNetUpConcatConv2d, self).__init__()

        if use_conv:
            self.upsample = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = UNetConv2d(in_size, out_size)

    def forward(self, input1, input2):
        output1 = self.upsample(input1)

        # Past solution
        # offset = output1.size()[2] - input2.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # output2 = F.pad(input2, padding)

        # Maybe good one
        diff_y = input2.size()[2] - output1.size()[2]
        diff_x = input2.size()[3] - output1.size()[3]

        output1 = F.pad(output1, [diff_x // 2, diff_x - diff_x // 2,
                                  diff_y // 2, diff_y - diff_y // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        return self.conv(torch.cat((input2, output1), dim=1))
