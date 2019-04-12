import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from models.common import get_upsampling_weight


class FCN16(nn.Module):

    def __init__(self,
                 num_classes=1,
                 use_pretrained=False,
                 learn_upconv=False):
        super(FCN16, self).__init__()
        self.learn_upconv = learn_upconv
        self.num_classes = num_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.num_classes, kernel_size=1)
        )

        self.score_conv4 = nn.Conv2d(512, self.num_classes, kernel_size=1)

        if self.learn_upconv:
            self.upscore2 = nn.ConvTranspose2d(
                self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False
            )
            self.upscore2.weight.data.copy_(get_upsampling_weight(self.num_classes, self.num_classes, 4))

            self.upscore16 = nn.ConvTranspose2d(
                self.num_classes, self.num_classes, kernel_size=32, stride=16, bias=False
            )
            self.upscore16.weight.data.copy_(get_upsampling_weight(self.num_classes, self.num_classes, 32))

        if use_pretrained:
            self._init_weights_vgg16()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x_size = x.size()
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.learn_upconv:
            upscore2 = self.upscore2(score)
            score_conv4c = self.score_conv4(conv4)[
                :, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])
            ]

            out = self.upscore16(score_conv4c + upscore2)[
                :, :, 27:(27 + x_size[2]), 27:(27 + x_size[3])
            ]

            return out.contiguous()
        else:
            score_conv4 = self.score_conv4(conv4)

            score = F.interpolate(score, size=score_conv4.size()[2:])
            score += score_conv4
            out = F.interpolate(score, size=x.size()[2:])

            return out

    def _init_weights_vgg16(self, copy_last_layer=False):
        vgg16 = models.vgg16(pretrained=True)

        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for vgg_layer, fcn_layer in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(vgg_layer, nn.Conv2d) and isinstance(fcn_layer, nn.Conv2d):
                    assert fcn_layer.weight.size() == vgg_layer.weight.size()
                    assert fcn_layer.bias.size() == vgg_layer.bias.size()
                    fcn_layer.weight.data = vgg_layer.weight.data
                    fcn_layer.bias.data = vgg_layer.bias.data

        for idx in [0, 3]:
            vgg_layer = vgg16.classifier[idx]
            fcn_layer = self.classifier[idx]

            fcn_layer.weight.data = vgg_layer.weight.data.view(fcn_layer.weight.size())
            fcn_layer.bias.data = vgg_layer.bias.data.view(fcn_layer.bias.size())

        if copy_last_layer:
            vgg_layer = vgg16.classifier[6]
            fcn_layer = self.classifier[6]

            fcn_layer.weight.data = vgg_layer.weight.data[:self.num_classes, :].view(fcn_layer.weight.size())
            fcn_layer.bias.data = vgg_layer.weight.data[:self.num_classes]

    def __repr__(self):
        return 'FCN16'
