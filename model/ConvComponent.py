import numpy as np
import torch.nn as nn
import torch
from .common_component import GDN

#Codes are implemented based on the next paper
#[1] E. Bourtsoulatze, D. B. Kurka, and D. G¡§und¡§uz, "Deep joint source channel coding for wireless image transmission,¡± 
#IEEE Trans. on Cogn.Commun. Netw., vol. 5, no. 3, pp. 567-579, May 2019.



def conv1(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv1(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)

class conv_block1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block1, self).__init__()
        self.conv = conv1(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.conv(x)
        out = self.prelu(out)
        return out

class deconv_block1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block1, self).__init__()
        self.deconv = deconv1(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.deconv(x)
        out = self.prelu(out)
        return out




