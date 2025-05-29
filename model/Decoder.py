from .common_component import *
from .ConvComponent import *
from .ResConvComponent import  deconv_ResBlock
from .SWComponent import SWGroup
from .FAComponent import FAGroup
from .ROIComponent import ROIGroup

class ConvDecoder(nn.Module):
    def __init__(self, model_info):
        super(ConvDecoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_stage = 2
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
        ksize = 5
        padding_L = (ksize-1)//2
        
        self.layer1 = deconv_block1(C, n_feats_list[-1], kernel_size=ksize, stride=1, padding=padding_L)
        self.layer2 = deconv_block1(n_feats_list[-1], n_feats_list[-2], kernel_size=ksize, stride=1, padding=padding_L)
        self.layer3 = deconv_block1(n_feats_list[-2], n_feats_list[-3], kernel_size=ksize, stride=1, padding=padding_L)
        self.layer4 = deconv_block1(n_feats_list[-3], n_feats_list[-4], kernel_size=ksize, stride=2, padding=padding_L,output_padding = 1)
        self.layer5 = deconv1(n_feats_list[-4], color_channel, kernel_size=ksize, stride=2, padding=padding_L,output_padding = 1)
        
    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out



class ResDecoder(nn.Module):
    def __init__(self, model_info):
        super(ResDecoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_stage = 2
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
        ksize = 5
        padding_L = (ksize-1)//2
        
        self.layer1 = deconv_ResBlock(C, n_feats_list[-1], use_deconv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)
        self.layer2 = deconv_ResBlock(n_feats_list[-1], n_feats_list[-2], use_deconv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)
        self.layer3 = deconv_ResBlock(n_feats_list[-2], n_feats_list[-3], use_deconv1x1=True, kernel_size=ksize, stride=1, padding=padding_L)
        self.layer4 = deconv_ResBlock(n_feats_list[-3], n_feats_list[-4], use_deconv1x1=True, kernel_size=ksize, stride=2, padding=padding_L,output_padding = 1)
        self.layer5 = deconv_ResBlock(n_feats_list[-4], color_channel, use_deconv1x1=True, kernel_size=ksize, stride=2, padding=padding_L,output_padding = 1)

        
    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out, 'None')

        return out




class SwinDecoder(nn.Module):
    def __init__(self, model_info,C=None):
        super(SwinDecoder, self).__init__()
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
        self.group_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        self.head_list = nn.Conv2d(C,n_feats_list[-1], kernel_size=1, stride=1, padding=0)
        
        for i in range(n_stage):
            group_layer = SWGroup(n_feats_list[-1-i],n_block_list[-1-i],num_heads_list[-1-i],window_size_list[-1-i],
            input_resolution=(input_resolution[0]//(2**(n_stage-i)),input_resolution[1]//(2**(n_stage-i))))
            self.group_layers.append(group_layer)
            
            if i==n_stage-1:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=color_channel)
            else:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=n_feats_list[-2-i])
            self.upsample_layers.append(upsample_layer)

        
    def forward(self, x, mask=None):
        out = self.head_list(x)
        for i in range(self.n_stage):            
            out = self.group_layers[i](out)
            out = self.upsample_layers[i](out)
        return out

class FADecoder(nn.Module):
    def __init__(self, model_info,C=None):
        super(FADecoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_block_list = model_info['n_block_list']
        window_size_list = model_info['window_size_list']
        ratio = model_info['ratio2']
        self.ratio = ratio
        n_stage = len(n_block_list)
        self.n_stage = n_stage
        rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        if not C:
            C = int(3*2*(2**n_stage)*(2**n_stage)*(1/rcpp))
            
        self.group_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        self.head_list = nn.Conv2d(C,n_feats_list[-1], kernel_size=1, stride=1, padding=0)
        
        for i in range(n_stage):
            group_layer = FAGroup(n_feats_list[-1-i],n_block_list[-1-i],window_size_list[-1-i],ratio)
            self.group_layers.append(group_layer)
            
            if i==n_stage-1:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=color_channel)
            else:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=n_feats_list[-2-i])
            self.upsample_layers.append(upsample_layer)

        
    def forward(self, x, mask=None):
        decision = []
        out = self.head_list(x)
        
        if self.training:
            for i in range(self.n_stage):
                out, mask = self.group_layers[i](out)
                out = self.upsample_layers[i](out)
                decision.extend(mask)
            return out, decision        
        else:
            for i in range(self.n_stage):            
                out = self.group_layers[i](out)
                out = self.upsample_layers[i](out)
            return out




class ROIDecoder(nn.Module):
    def __init__(self, model_info,C):
        super(ROIDecoder, self).__init__()
        color_channel = model_info['color_channel']
        n_feats_list = model_info['n_feats_list']
        n_block_list = model_info['n_block_list']
        window_size_list = model_info['window_size_list']
        ratio = model_info['ratio2']
        self.ratio = ratio
        n_stage = len(n_block_list)
        self.n_stage = n_stage
        self.group_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        self.head_list = nn.Conv2d(C,n_feats_list[-1], kernel_size=1, stride=1, padding=0)
        
        for i in range(n_stage):
            group_layer = ROIGroup(n_feats_list[-1-i],n_block_list[-1-i],window_size_list[-1-i],ratio)
            self.group_layers.append(group_layer)
            
            if i==n_stage-1:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=color_channel)
            else:
                upsample_layer = PatchReverseMerging(dim=n_feats_list[-1-i],out_dim=n_feats_list[-2-i])
            self.upsample_layers.append(upsample_layer)

        
    def forward(self, x, mask):
        out = self.head_list(x)
        for i in range(self.n_stage):            
            out = self.group_layers[i](out, mask)
            out = self.upsample_layers[i](out)
        return out






