from .common_component import *
from .ConvComponent import *
from .ResConvComponent import conv_ResBlock
from .SWComponent import SWGroup
from .FAComponent import FAGroup
from .ROIComponent import ROIGroup

class ConvEncoder(nn.Module):
    def __init__(self, model_info):
        super(ConvEncoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_stage = 2
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
        ksize = 5
        padding_L = (ksize-1)//2
        
        self.layer1 = conv_block1(color_channel, n_feats_list[0], kernel_size=ksize, stride=2, padding=padding_L)       
        self.layer2 = conv_block1(n_feats_list[0], n_feats_list[1], kernel_size=ksize, stride=2, padding=padding_L)        
        self.layer3 = conv_block1(n_feats_list[1], n_feats_list[2], kernel_size=ksize, stride=1, padding=padding_L)
        self.layer4 = conv_block1(n_feats_list[2], n_feats_list[3], kernel_size=ksize, stride=1, padding=padding_L)
        self.layer5 = conv1(n_feats_list[3], C, kernel_size=ksize, stride=1, padding=padding_L)

        
    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out

class ResEncoder(nn.Module):
    def __init__(self, model_info):
        super(ResEncoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_stage = 2
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
        ksize = 5
        padding_L = (ksize-1)//2
        
        self.layer1 = conv_ResBlock(color_channel, n_feats_list[0], use_conv1x1=True, kernel_size=ksize, stride=2, padding=padding_L)       
        self.layer2 = conv_ResBlock(n_feats_list[0], n_feats_list[1], use_conv1x1=True, kernel_size=ksize, stride=2, padding=padding_L)          
        self.layer3 = conv_ResBlock(n_feats_list[1], n_feats_list[2], use_conv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)  
        self.layer4 = conv_ResBlock(n_feats_list[2], n_feats_list[3], use_conv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)  
        self.layer5 = conv_ResBlock(n_feats_list[3], C, use_conv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)

        
    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out,skip=1)

        return out


        
class SwinEncoder(nn.Module):
    def __init__(self, model_info,C=None):
        super(SwinEncoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_block_list = model_info['n_block_list']
        num_heads_list = model_info['num_heads_list']
        window_size_list = model_info['window_size_list']
        input_resolution = model_info['input_resolution']
        n_stage = len(n_block_list)
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        if not C:
            C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
        self.downsample_layers = nn.ModuleList()
        self.group_layers = nn.ModuleList()
        
        for i in range(n_stage):
            if i==0:
                downsample_layer = PatchEmbed(patch_size=2, in_chans=color_channel, embed_dim=n_feats_list[i])
            else:
                downsample_layer = PatchMerging(dim=n_feats_list[i-1],out_dim=n_feats_list[i])
            self.downsample_layers.append(downsample_layer)
            
            group_layer = SWGroup(n_feats_list[i],n_block_list[i],num_heads_list[i],window_size_list[i],
            input_resolution=(input_resolution[0]//(2**(i+1)),input_resolution[1]//(2**(i+1))))
            self.group_layers.append(group_layer)
            
        self.head_list = nn.Conv2d(n_feats_list[-1], C, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, mask=None):
        out = x
        for i in range(self.n_stage):
            out = self.downsample_layers[i](out)
            out = self.group_layers[i](out)
        out = self.head_list(out)
        return out        
        
        
class FAEncoder(nn.Module):
    def __init__(self, model_info,C=None):
        super(FAEncoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_block_list = model_info['n_block_list']
        window_size_list = model_info['window_size_list']
        ratio = model_info['ratio1']
        self.ratio = ratio
        n_stage = len(n_block_list)
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        if not C:
            C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))

        self.downsample_layers = nn.ModuleList()
        self.group_layers = nn.ModuleList()
        
        for i in range(n_stage):
            if i==0:
                downsample_layer = PatchEmbed(patch_size=2, in_chans=color_channel, embed_dim=n_feats_list[i])
            else:
                downsample_layer = PatchMerging(dim=n_feats_list[i-1],out_dim=n_feats_list[i])
            self.downsample_layers.append(downsample_layer)
            
            group_layer = FAGroup(n_feats_list[i],n_block_list[i],window_size_list[i],ratio)
            self.group_layers.append(group_layer)
            
        self.head_list = nn.Conv2d(n_feats_list[-1], C, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, mask=None):
        out = x
        decision = []
                
        if self.training:
            for i in range(self.n_stage):
                out = self.downsample_layers[i](out)
                out, mask  = self.group_layers[i](out)
                decision.extend(mask)
            out = self.head_list(out)
            return out, decision
        else:
            for i in range(self.n_stage):
                out = self.downsample_layers[i](out)
                out = self.group_layers[i](out)
            out = self.head_list(out)
            return out
             


class ROIEncoder(nn.Module):
    def __init__(self, model_info,C):
        super(ROIEncoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_block_list = model_info['n_block_list']
        window_size_list = model_info['window_size_list']
        ratio = model_info['ratio1']
        self.ratio = ratio
        n_stage = len(n_block_list)
        self.n_stage = n_stage
        self.downsample_layers = nn.ModuleList()
        self.group_layers = nn.ModuleList()
        
        for i in range(n_stage):
            if i==0:
                downsample_layer = PatchEmbed(patch_size=2, in_chans=color_channel, embed_dim=n_feats_list[i])
            else:
                downsample_layer = PatchMerging(dim=n_feats_list[i-1],out_dim=n_feats_list[i])
            self.downsample_layers.append(downsample_layer)
            
            group_layer = ROIGroup(n_feats_list[i],n_block_list[i],window_size_list[i],ratio)
            self.group_layers.append(group_layer)
            
        self.head_list = nn.Conv2d(n_feats_list[-1], C, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, mask):
        out = x
        for i in range(self.n_stage):
            out = self.downsample_layers[i](out)
            out = self.group_layers[i](out, mask)
        out = self.head_list(out)
        return out



























