import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair

from bayesian.distributions import Normal, Normalout, distribution_selector

cuda = torch.cuda.is_available()


class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


# noinspection PyAbstractClass
class _ConvNd(nn.Module):
    """
    Describes a Bayesian convolutional layer with
    a distribution over each of the weights and biases
    in the layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, p_logvar_init=0.05, p_pi=1.0, q_logvar_init=0.05):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        # initialize log variance of p and q
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = math.log(q_logvar_init)

        # ...and output...
        self.conv_qw_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.conv_qw_std = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        # initialize all parameters
        self.reset_parameters()

    def reset_parameters(self):
        # initialise (learnable) approximate posterior parameters
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.conv_qw_mean.data.uniform_(-stdv, stdv)
        self.conv_qw_std.data.fill_(self.q_logvar_init)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BBBConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(BBBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, _pair(0),
                                        groups)

    def forward(self, x):
        raise NotImplementedError()

    def convprobforward(self, x):
        """
        Convolutional probabilistic forwarding method.
        :param x: data tensor
        :return: output, KL-divergence
        """

        sigma_weight = torch.exp(self.conv_qw_std)

        weight = self.conv_qw_mean + sigma_weight * self.eps_weight.normal_()

        output = F.conv2d(input=x, weight=weight, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)

        kl = math.log(self.p_logvar_init) - self.conv_qw_std + (sigma_weight ** 2 + self.conv_qw_mean ** 2) / (
                2 * self.p_logvar_init ** 2) - 0.5
        kl = kl.sum()

        return output, kl


class GaussianVariationalInference(nn.Module):

    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        logpy = -self.loss(logits, y)

        ll = logpy - beta * kl
        loss = -ll

        return loss
