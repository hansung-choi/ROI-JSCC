import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
from torch.autograd import Function
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from bisect import bisect
import torch.nn.functional as F
import numpy as np
from einops import rearrange

"""
Codes are modified from the next references for baseline comparison of our research

GDN
[1] Zhang W, Zhang H, Ma H, et al. Predictive and Adaptive Deep Coding for Wireless Image Transmission 
in Semantic Communication. IEEE Transactions on Wireless Communications, 2023.

PatchMerging, PatchReverseMerging, PatchEmbed, Channel
[2] Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping (2024). 
SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding. 
IEEE Transactions on Cognitive Communications and Networking.

"""


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B,C,H,W = x.shape
        x = rearrange(x,'b c h w -> b (h w) c', h=H,w=W)
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x,'b (h w) c -> b c h w', h=H//2,w=W//2)

        return x



class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B,C,H,W = x.shape
        x = rearrange(x,'b c h w -> b (h w) c', h=H,w=W)
        B, L, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        #x = x.flatten(2).permute(0, 2, 1)
        #x = rearrange(x,'b (h w) c -> b c h w', h=H*2,w=W*2)
        return x



class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=60, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = rearrange(x,'b (h w) c -> b c h w', h=H//self.patch_size[0],w=W//self.patch_size[1])
        return x


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, chan_type='AWGN'):
        super(Channel, self).__init__()

        self.chan_type = chan_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h = torch.sqrt(torch.randn(1) ** 2
                                + torch.randn(1) ** 2) / 1.414


    def gaussian_noise_layer(self, input_layer, std, name=None):
        #print('AWGN')
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        #print('Rayleigh')
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
                       
        noise = noise.to(input_layer.get_device())
        h = h.to(input_layer.get_device())
        return input_layer * h + noise
 

    def complex_normalize(self, x): #imagewise normalize, attention when using masking operation (making return 1d tensor).
        # <y,y> / k <= 1        
        x_dim = x.size()
        
        x = x.reshape((x_dim[0],-1))
        out = F.normalize(x,dim=1)

        out = out * torch.sqrt(torch.tensor(x.size()[1],dtype=torch.float32, device=x.device))
        out = out * torch.sqrt(torch.tensor(0.5, device=x.device))
        out = out.reshape(x_dim)
        
        return out

     
    def forward(self, input, chan_param): # chan_param: {SNR} dB
        channel_tx = self.complex_normalize(input)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        return channel_output

    def complex_forward(self, channel_in, chan_param): # chan_param: {SNR} dB
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'AWGN':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'Rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            return chan_output


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx



def spatial_mask_upsample(x,H,W):
    B,h,w = x.size()
    
    assert H%h == 0, "target height should be multiple of input height"
    assert W%w == 0, "target width should be multiple of input width"
    mask = x.reshape(B,h,w,1).repeat(1,1,1,int(H//h * W//w))
    mask = rearrange(mask,'B h w (dh dw) -> B (h dh) (w dw)',dh=int(H//h),dw=int(W//w))
    return mask

def spatial_mask_downsample(x,h,w):
    B,H,W = x.size()
    
    assert H%h == 0., "input height should be multiple of target height"
    assert W%w == 0., "input width should be multiple of target width"
        
    mask = x.reshape(B,1,H,W)
    mask = F.adaptive_avg_pool2d(mask,(h,w))
    mask = mask.reshape(B,h,w)

    return mask

def spatial_mask_fit(x,H,W):
    B,h,w = x.size()
    if h==H and w==W:
        return x
    
    
    if h <=H and w <=W:
        mask = spatial_mask_upsample(x,H,W)
    elif h > H and w > W:
        mask = spatial_mask_downsample(x,H,W)
    else:
        raise ValueError(f'{H}, {W} both should be larger or smaller than its corresponding spatial dimension of x')
    
    return mask
    

def refine_spatial_mask_list(spatial_mask_list,images):
    B,C,H,W = images.size()
    new_spatial_mask_list = []
    
    for _, mask in enumerate(spatial_mask_list):
        new_mask = spatial_mask_fit(mask,H,W)
        new_spatial_mask_list.append(new_mask)
    
    return new_spatial_mask_list


def get_spatial_mask_attention11(spatial_mask_list):
    new_spatial_mask_list = []
    for _, mask in enumerate(spatial_mask_list):
        B, H, W = mask.size()
        new_mask = mask.reshape(B,H,W,1).float()
        new_spatial_mask_list.append(new_mask)

    new_spatial_mask = torch.cat(new_spatial_mask_list,dim=3) #new_spatial_mask.size() = B X H X W X len(spatial_mask_list)
    new_spatial_mask = torch.mean(new_spatial_mask, dim=3) #new_spatial_mask.size() = B X H X W
    new_spatial_mask = new_spatial_mask.reshape(B,1,H,W) #new_spatial_mask.size() = B X 1 X H X W
    
    B, _, H, W = new_spatial_mask.size()
    
    attention_mask = torch.zeros_like(new_spatial_mask)    
    new_spatial_mask_mean = torch.mean(new_spatial_mask, dim=(1,2,3))
    new_spatial_mask_mean = new_spatial_mask_mean.reshape(B,1,1,1).repeat(1,1,H,W)

    attention_mask[new_spatial_mask>=new_spatial_mask_mean] = 1.
    attention_mask[new_spatial_mask<new_spatial_mask_mean] = 0.
    
    return attention_mask #attention_mask.size() = B X 1 X H X W


def get_spatial_mask_attention(spatial_mask_list):
    new_spatial_mask_list = []
    for _, mask in enumerate(spatial_mask_list):
        B, H, W = mask.size()
        new_mask = mask.reshape(B,H,W).float()
        new_spatial_mask_list.append(new_mask)
    #torch.stack
    #new_spatial_mask = torch.cat(new_spatial_mask_list,dim=1) #new_spatial_mask.size() = B X len(spatial_mask_list) X H X W
    new_spatial_mask = torch.stack(new_spatial_mask_list,dim=1) #new_spatial_mask.size() = B X len(spatial_mask_list) X H X W
    
    
    new_spatial_mask = torch.mean(new_spatial_mask, keepdim=True, dim=1) #new_spatial_mask.size() = B X 1 X H X W
    
    B, _, H, W = new_spatial_mask.size()
    
    attention_mask = torch.zeros_like(new_spatial_mask)    
    new_spatial_mask_mean = torch.mean(new_spatial_mask, dim=(1,2,3))
    new_spatial_mask_mean = new_spatial_mask_mean.reshape(B,1,1,1).repeat(1,1,H,W)
    
    #attention_mask[new_spatial_mask>=new_spatial_mask_mean] = 1.
    #attention_mask[new_spatial_mask<new_spatial_mask_mean] = 0.



    new_spatial_mask_median, _ = torch.median(new_spatial_mask, 2)
    new_spatial_mask_median, _ = torch.median(new_spatial_mask_median, 2)
    new_spatial_mask_median = new_spatial_mask_median.reshape(B,1,1,1).repeat(1,1,H,W)    
    
    attention_mask[new_spatial_mask>=new_spatial_mask_median] = 1.
    attention_mask[new_spatial_mask<new_spatial_mask_median] = 0.
    
    return attention_mask #attention_mask.size() = B X 1 X H X W

def get_focused_images(spatial_mask_list,images):
    new_spatial_mask_list = refine_spatial_mask_list(spatial_mask_list,images)
    
    attention_mask = get_spatial_mask_attention(new_spatial_mask_list)
    focused_image = images * attention_mask
    
    return focused_image

def _2DIndex_to_1DIndex(h,w,H,W):
    return h*W+w
    
def _1DIndex_to_2DIndex(i,H,W):
    return i//W, i%W

    
def get_ROI_attention_mask(ROI_Index,B,H,W):
    Ih,Iw = ROI_Index #Ih: height of interest index, Iw: width of interest index
    
    ROI_attention_mask = torch.zeros(B,H,W)
    ROI_attention_mask[:,Ih,Iw]= 1.
    return ROI_attention_mask
    
def get_peripheral_2D_Index_list(ROI_Index,H,W):
    Ih,Iw = ROI_Index
    peripheral_2D_Index_list = []
    peripheral_range = [-1,0,1]
    
    for dh in peripheral_range:
        for dw in peripheral_range:
            if Ih+dh >=0 and Ih+dh <H and Iw+dw >=0 and  Iw+dw < W:
                peripheral_2D_Index_list.append([Ih+dh,Iw+dw])
    peripheral_2D_Index_list.remove([Ih,Iw])
    
    return peripheral_2D_Index_list
                

   
def get_peripheral_attention_mask(ROI_Index,B,H,W):
    Ih,Iw = ROI_Index #Ih: height of interest index, Iw: width of interest index
    
    peripheral_attention_mask = torch.zeros(B,H,W)
    peripheral_2D_Index_list = get_peripheral_2D_Index_list(ROI_Index,H,W)
    for peripheral_2D_Index in peripheral_2D_Index_list:
        h,w = peripheral_2D_Index
        peripheral_attention_mask[:,h,w] = 1.    
    
    return peripheral_attention_mask
    
def get_weighted_total_attention_mask(ROI_Index,B,H,W):

    ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,H,W)
    peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,H,W)
    weighted_total_attention_mask = ROI_attention_mask+0.5*peripheral_attention_mask
    return weighted_total_attention_mask
    

def get_RONI_attention_mask(ROI_Index,B,H,W):
    Ih,Iw = ROI_Index #Ih: height of interest index, Iw: width of interest index
    
    peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,H,W)
    ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,H,W)
    
    total_attention = peripheral_attention_mask + ROI_attention_mask
    
    RONI_attention_mask = torch.ones(B,H,W)
    RONI_attention_mask = RONI_attention_mask- total_attention
    
    return RONI_attention_mask




def get_channelwise_valid_z_mask(z,rate_info): #z: latent_feature_of_channel, rate_info:[start_feature_channel, end_feature_channel]  of valid feature   
    z_dim = z.size()
    mask = torch.zeros_like(z)  #.bool()
    if len(z_dim) ==3: #when z is B X L X C tensor
        mask[:,:,rate_info[0]:rate_info[1]] = 1.
        mask = mask.bool()
        valid_z = torch.masked_select(z, mask).reshape(z_dim[0],z_dim[1],-1)
    elif len(z_dim) ==4: #when z is B X C X H X W tensor
        mask[:,rate_info[0]:rate_info[1],:,:] = 1.
        mask = mask.bool()
        valid_z = torch.masked_select(z, mask).reshape(z_dim[0],-1,z_dim[2],z_dim[3])
    else:
        raise ValueError(f'shape for{z_dim} (len(z_dim)) is not implemented yet')
    
    return valid_z, mask

def padding_received_z_hat(z,z_hat,mask):
    received_z_hat = torch.zeros_like(z)
    received_z_hat[mask] = z_hat.reshape(-1)
    return received_z_hat

def patch_wise_calculation(batch_patch_image_tensor_hat,batch_patch_image_tensor,image_wise_criterion): 
    #batch_patch_image_tensor: B X d^2 X C X H/d X W/d
    #image_wise_criterion: return criterion results for image_wise
    input_dim = batch_patch_image_tensor.size()
    patch_image_tensors_hat = batch_patch_image_tensor_hat.reshape(-1,input_dim[2],input_dim[3],input_dim[4])
    patch_image_tensors = batch_patch_image_tensor.reshape(-1,input_dim[2],input_dim[3],input_dim[4])

    patch_wise_calculation_result = image_wise_criterion(patch_image_tensors_hat,patch_image_tensors)
    patch_wise_calculation_result = patch_wise_calculation_result.reshape(input_dim[0],input_dim[1])
    #print("patch_wise_calculation_result:",patch_wise_calculation_result)
    return patch_wise_calculation_result # It is criterion result for each patch. Dimension is B X d^2.

    
def patch_division(image_tensor,d):
    B, C, H, W = image_tensor.shape
    d = int(d)
    batch_patch_image_tensor = rearrange(image_tensor,'b c (h dh) (w dw) -> b (h w) c dh dw', b=B,c=C,h=d,w=d)    
    return batch_patch_image_tensor


# B X d^2 X C X H/d X W/d -> B X C X H X W (reverse patch division), B should be larger than 1.      
def reverse_patch_division(batch_patch_image_tensor):
    B, P, C, h, w = batch_patch_image_tensor.size()
    d = int(np.sqrt(P))    
    image_tensor = rearrange(batch_patch_image_tensor,'b (h w) c dh dw -> b c (h dh) (w dw)', b=B ,h=d,w=d)     
    return image_tensor   

     
    
    
def patch_rate_allocation(mask, channel_rate_list):
    #we assume # of patches = 16
    #patch_wise_psnr: B X d^2, channel_rate_list: list of possible patchwise channel rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patch_wise_importance = mask.clone().detach()
    patch_wise_importance = rearrange(patch_wise_importance,'b h w -> b (h w)')
    
    patch_wise_importance = patch_wise_importance.clone().detach().to(device)
    num_patch = patch_wise_importance.size()[1]
    sorted_patch_wise_importance, sort_index = patch_wise_importance.sort(dim=-1,descending=False)
    
    # the number of patch for each rate allocation
    rate_wise_patch_num_list = [7,8,1]
    
    # the cumulative number of patches for rate allocation
    rate_allocation_standard_list = [sum(rate_wise_patch_num_list[:i+1]) for i in range(len(rate_wise_patch_num_list))]
    
    patch_wise_rate = torch.zeros_like(patch_wise_importance).long()

    
    for rate_num in range(len(channel_rate_list)):
        if rate_num ==0:
            interest_sort_index = sort_index[:,:rate_allocation_standard_list[rate_num]]
        else:
            interest_sort_index = sort_index[:,rate_allocation_standard_list[rate_num-1]:rate_allocation_standard_list[rate_num]]
        for i in range(len(interest_sort_index)):
            patch_wise_rate[i,interest_sort_index[i]] = channel_rate_list[rate_num]
    
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    #print("patch_wise_rate.float().mean(dim=-1):",patch_wise_rate.float().mean(dim=-1))
    
    return patch_wise_rate

       
def get_patchwise_valid_z_mask(batch_patch_image_tensor,patch_wise_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch_patch_image_tensor: B X d^2 X C X H/d X W/d
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    z = batch_patch_image_tensor   
    B, P, C, h, w = z.size()
    #.bool()

    z_BPLC = z.flatten(3).permute(0,1,3,2) # B X P X L X C where P=d^2, L=H/d X W/d
    valid_mask = torch.zeros_like(z_BPLC)
    max_rate = C
    mask = torch.arange(0, max_rate).repeat(B, P, h*w, 1).to(device)
    extended_patch_wise_rate = patch_wise_rate.reshape(B, P, 1, 1).repeat(1,1,h*w, max_rate)

    
    valid_mask[mask<extended_patch_wise_rate] = 1.
    valid_mask[mask>=extended_patch_wise_rate] = 0.
    valid_mask = valid_mask.bool()
    valid_z_BPLC = torch.masked_select(z_BPLC, valid_mask).reshape(B,-1)

    return z_BPLC, valid_z_BPLC, valid_mask
    
def get_valid_z(z,mask): #z: latent_feature_of_channel, mask: info of valid feature
    B = z.size()[0]
    valid_z = torch.masked_select(z, mask).reshape(B,-1) # shape should be B X -1 due to power allocation of channel.
    
    return valid_z
    
#BLC -> BCHW -> BPCHW -> BPLC   
    
def BLC_to_BCHW(z_BLC,H,W):
    z_BLC_dim = z_BLC.size()
    B, L, C = z_BLC_dim
    assert L == H * W, "input feature has wrong size"   
    z_BCHW = z_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return z_BCHW
    
    
def BCHW_to_BLC(z_BCHW):
    z_BCHW_dim = z_BCHW.size()
    B, C,H,W = z_BCHW_dim
    z_BLC = z_BCHW.flatten(2).permute(0, 2, 1)
    return z_BLC
    

def BPLC_to_BPCHW(z_BPLC,H,W):    
    z_BPLC_dim = z_BPLC.size()
    B, P, L, C = z_BPLC_dim
    assert L == H * W, "input feature has wrong size"   
    z_BPCHW = z_BPLC.reshape(B, P, H, W, C).permute(0, 1, 4, 2, 3)
    return z_BPCHW
    
    
def BPCHW_to_BPLC(z_BPCHW):    
    z_BPCHW_dim = z_BPCHW.size()
    B, P, C, H,W = z_BPCHW_dim  
    z_BPLC = z_BPCHW.flatten(3).permute(0,1, 3, 2)
    return z_BPLC    



if __name__ == '__main__':
    x = [1.,2.,3.,4.]
    x = torch.tensor(x)
    print(x[0:1])
    x = x.reshape(1,2,2)
    
    Y = spatial_mask_upsample(x,8,8)
    y = spatial_mask_downsample(Y,2,2)
    print("x:",x)
    print("Y:",Y)    
    print("y:",y)    
    
















