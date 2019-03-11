import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu2D(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super(ConvBnRelu2D, self).__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.batch_norm:
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
