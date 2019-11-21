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
        self.q_logvar_init = q_logvar_init

        # approximate posterior weights...
        self.qw_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.qw_logvar = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        # ...and output...
        self.conv_qw_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.conv_qw_si = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        # ...as normal distributions
        self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        self.conv_qw = Normalout(mu=self.conv_qw_mean, si=self.conv_qw_si)

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        # (does not have any trainable parameters so we use fixed normal or fixed mixture normal distributions)
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all parameters
        self.reset_parameters()

    def reset_parameters(self):
        # initialise (learnable) approximate posterior parameters
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.conv_qw_mean.data.uniform_(-stdv, stdv)
        self.conv_qw_si.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

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
        return self.convprobforward(x)

    def convprobforward(self, x):
        """
        Convolutional probabilistic forwarding method.
        :param x: data tensor
        :return: output, KL-divergence
        """

        # local reparameterization trick for convolutional layer
        conv_qw_mean = F.conv2d(input=x, weight=self.qw_mean, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
        conv_qw_si = torch.sqrt(
            1e-8 + F.conv2d(input=x.pow(2), weight=torch.exp(self.log_alpha) * self.qw_mean.pow(2),
                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))

        if cuda:
            conv_qw_mean.cuda()
            conv_qw_si.cuda()

            output = conv_qw_mean + conv_qw_si * (torch.randn(conv_qw_mean.size())).cuda()

            output.cuda()

        else:
            output = conv_qw_mean + conv_qw_si * (torch.randn(conv_qw_mean.size()))

        w_sample = self.conv_qw.sample()

        # KL divergence
        qw_logpdf = self.conv_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

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
