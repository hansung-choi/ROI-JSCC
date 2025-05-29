import numpy as np
import torch.nn as nn
import torch
from .common_component import GDN


#Codes are modified from the next references for baseline comparison of our research
#[1] W. Zhang, H. Zhang, H. Ma, H. Shao, N. Wang, and V. C. Leung, ¡°Predictive and adaptive deep coding for wireless image transmission
#in semantic communication,¡± IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5486-5501, Jan. 2023.





def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gdn = GDN(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.conv(x)
        out = self.gdn(out)
        out = self.prelu(out)
        return out

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block, self).__init__()
        self.deconv = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.gdn = nn.GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv(x)
        out = self.gdn(out)
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        elif activate_func=='None':
            out = out
        return out   
    
class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr):
        # snr : dB, scalar or B dim vector
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        if snr.shape[0]>1:
            snr = snr.squeeze()
        snr = snr.unsqueeze(1)  
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out


class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    def forward(self, x,skip=None): 
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out+x
        if skip:
            out = out
        else:
            out = self.prelu(out)
        return out 

class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out+x
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        elif activate_func=='None':
            out = out
        return out 
        
class ResGroup(nn.Module):
    def __init__(self, n_feats, n_block, ksize=5):
        super(ResGroup, self).__init__()
        padding_L = (ksize-1)//2        
        self.body = nn.ModuleList([conv_ResBlock(n_feats,n_feats,kernel_size=ksize,stride=1,padding=padding_L) for i in range(n_block)])
        
    def forward(self, x):
        # Input: B X C X H X W
        # Output: B X C X H X W
        for _, blk in enumerate(self.body):
            x = blk(x)
        return x

class conv_ResAFBlock(nn.Module):
    def __init__(self, n_feats, ksize=5):
        super(conv_ResAFBlock, self).__init__()
        padding_L = (ksize-1)//2
        n_hfeats = n_feats//2
        self.block1 = conv_ResBlock(n_feats,n_feats,kernel_size=ksize,stride=1,padding=padding_L)
        self.block2 = AF_block(n_feats, n_hfeats, n_feats)
        
    def forward(self, x, snr): 
        out = self.block1(x)
        out = self.block2(out, snr)
        return out 


class ResAFGroup(nn.Module):
    def __init__(self, n_feats, n_block, ksize=5):
        super(ResAFGroup, self).__init__()
        padding_L = (ksize-1)//2        
        self.body = nn.ModuleList([conv_ResAFBlock(n_feats,ksize=ksize) for i in range(n_block)])
        
    def forward(self, x, snr):
        # Input: B X C X H X W
        # Output: B X C X H X W
        for _, blk in enumerate(self.body):
            x = blk(x)
        return x
        

def mask_gen(N, cr, ch_max = 48):
    MASK = torch.zeros(cr.shape[0], N).int()
    nc = N//ch_max
    for i in range(0, cr.shape[0]):
        L_i = nc*torch.round(ch_max*cr[i]).int()
        MASK[i, 0:L_i] = 1
    return MASK
