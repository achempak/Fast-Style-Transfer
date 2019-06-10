import numpy as np
import torch
from torch import nn
from collections import OrderedDict 
from collections import namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm='instance'):
        super(ConvLayer, self).__init__()
        self.norm = norm
        padding = kernel_size//2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if self.norm is 'instance':
            self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        
    def forward(self, x):
        y = self.reflection_pad(x)
        y = self.conv(y)
        if self.norm is 'instance':
            y = self.instance_norm(y)
        return y
    
class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)
    
    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = y + residual
        return y
    
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, out_padding):
        super(DeconvLayer, self).__init__()
        padding = kernel_size//2
        self.frac_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                           padding, out_padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        y = self.frac_conv(x)
        y = self.instance_norm(y)
        return y
    
class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1',ConvLayer(3, 32, 9, 1)),
            ('relu1',nn.ReLU()),
            ('conv2',ConvLayer(32, 64, 3, 2)),
            ('relu2',nn.ReLU()),
            ('conv3',ConvLayer(64, 128, 3, 2)),
            ('relu3',nn.ReLU()),
            
            ('res1',ResidualLayer(128)),
            ('res2',ResidualLayer(128)),
            ('res3',ResidualLayer(128)),
            ('res4',ResidualLayer(128)),
            ('res5',ResidualLayer(128)),

            ('deconv1',DeconvLayer(128, 64, 3, 2, 1)),
            ('relu4',nn.ReLU()),
            ('deconv2',DeconvLayer(64, 32, 3, 2, 1)),
            ('relu5',nn.ReLU()),
            ('conv4',ConvLayer(32, 3, 9, 1, norm='none'))
        ]))
        
    def forward(self, x):
        y = self.model(x)
        q = nn.Tanh()
        return (q(y))
        #return y