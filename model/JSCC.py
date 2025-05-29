from .common_component import *
from .Encoder import *
from .Decoder import *
import random

class ConvJSCC(nn.Module):
    def __init__(self, model_info):
        super(ConvJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = 2
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp        
        self.chan_type = model_info['chan_type']

        self.Encoder = ConvEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = ConvDecoder(model_info)
        
    def forward(self, x,ROI_Index=(1,1), SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)
        decoder_input = self.channel(encoder_output,SNR_info)
        decoder_output = self.Decoder(decoder_input)

        return decoder_output
    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch

class ResJSCC(nn.Module):
    def __init__(self, model_info):
        super(ResJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = 2
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp        
        self.chan_type = model_info['chan_type']

        self.Encoder = ResEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = ResDecoder(model_info)
        
    def forward(self, x,ROI_Index=(1,1), SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)
        decoder_input = self.channel(encoder_output,SNR_info)
        decoder_output = self.Decoder(decoder_input)

        return decoder_output
    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class SwinJSCC(nn.Module):
    def __init__(self, model_info):
        super(SwinJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = SwinEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = SwinDecoder(model_info)
        
    def forward(self, x,ROI_Index=(1,1), SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)        
        decoder_input = self.channel(encoder_output,SNR_info)        
        decoder_output = self.Decoder(decoder_input)

        return decoder_output

    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class FAJSCC(nn.Module): # Feature Importance Aware JSCC
    def __init__(self, model_info):
        super(FAJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = FAEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = FADecoder(model_info)
        
    def forward(self, x,ROI_Index=(1,1), SNR_info=5):
        # input shape = B X C X H X W
        decision = []

        if self.training:
            encoder_output, mask = self.Encoder(x)
            decision.extend(mask)                    
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output, mask = self.Decoder(decoder_input)
            decision.extend(mask)
            return decoder_output, decision          

        else:
            encoder_output = self.Encoder(x)        
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output = self.Decoder(decoder_input)
            return decoder_output


    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch







class ROIJSCC(nn.Module): # Feature Importance Guide JSCC
    def __init__(self, model_info):
        super(ROIJSCC, self).__init__()
        #self.epoch = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.eta = model_info['eta']
        
        # we assume total image is divided with 16 patches
        decreased_RONI_resource = int(self.eta*self.C) #decrease the number of transmitted pixels at Non Region of Interest, 0.1, 0.2
        increased_ROI_resource = int(7*decreased_RONI_resource) #increase the number of transmitted pixels at Region of Interest
        RONI_C = self.C - decreased_RONI_resource
        ROI_C = self.C + increased_ROI_resource
        self.channel_rate_list = [RONI_C,self.C,ROI_C]
        
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = ROIEncoder(model_info,ROI_C)
        self.channel = Channel(self.chan_type)
        self.Decoder = ROIDecoder(model_info,ROI_C)
        
        self.d = 4
        

    def forward(self, x,ROI_Index=(1,1), SNR_info=5): 
        # Pathwise Rate allocation based on given patchwise rate
        # Important! do not change feature z shape of encoder output.
        # Make mask for rate allocation and power tensor and change their shape to be aligned with encoder output.
        # Repeat! do not change feature z shape of encoder output.
        # input shape = B X C X H X W
        # patch_wise_rate = B X d^2 where d^2 is the number of patches for each image
        # h = H/d, w = W/d (height and width of each patch)
        # z_BCHW -> z_BPChw -> valid_z_BPLC -> channel -> padded_z_BPLC_hat -> z_BPChw_hat -> z_BCHW_hat
        
        B, C, H, W = x.size()
        d = self.d
        attention_mask = get_weighted_total_attention_mask(ROI_Index,B,d,d).to(self.device)
        patch_wise_rate = patch_rate_allocation(attention_mask, self.channel_rate_list)
        #print("mask:",mask[0])
        #print("patch_wise_rate:",patch_wise_rate[0])
        #return None
        
        
        d = int(np.sqrt(patch_wise_rate.size()[1]))


        z_BCHW = self.Encoder(x,attention_mask)
        
        B, C, H, W = z_BCHW.size()
        h = H//d
        w = W//d
            
        z_BPChw = patch_division(z_BCHW,d)
        
        z_BPLC, valid_z_BPLC, mask_BPLC = get_patchwise_valid_z_mask(z_BPChw, patch_wise_rate)
        
        mask_BPCHW = BPLC_to_BPCHW(mask_BPLC,h,w)
        
        mask_BCHW = reverse_patch_division(mask_BPCHW)
        
        valid_z_BCHW = get_valid_z(z_BCHW,mask_BCHW)
               
        
        z_BCHW_hat = self.channel(valid_z_BCHW,SNR_info)
        
        padded_z_BCHW_hat = padding_received_z_hat(z_BCHW, z_BCHW_hat,mask_BCHW)
              
        task_output = self.Decoder(padded_z_BCHW_hat,attention_mask)

        return task_output


    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class FAJSCCwRB(nn.Module):  #FAJSCC, RB: ROI focusing bandwidth allocation, RL: ROI focusing loss
    def __init__(self, model_info):
        super(FAJSCCwRB, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.eta = model_info['eta']
        
        # we assume total image is divided with 16 patches
        decreased_RONI_resource = int(self.eta*self.C) #decrease the number of transmitted pixels at Non Region of Interest
        increased_ROI_resource = int(7*decreased_RONI_resource) #increase the number of transmitted pixels at Region of Interest
        RONI_C = self.C - decreased_RONI_resource
        ROI_C = self.C + increased_ROI_resource
        self.channel_rate_list = [RONI_C,self.C,ROI_C]
        
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = FAEncoder(model_info,ROI_C)
        self.channel = Channel(self.chan_type)
        self.Decoder = FADecoder(model_info,ROI_C)
        
        self.d = 4

    def forward(self, x,ROI_Index=(1,1), SNR_info=5): 
        # Pathwise Rate allocation based on given patchwise rate
        # Important! do not change feature z shape of encoder output.
        # Make mask for rate allocation and power tensor and change their shape to be aligned with encoder output.
        # Repeat! do not change feature z shape of encoder output.
        # input shape = B X C X H X W
        # patch_wise_rate = B X d^2 where d^2 is the number of patches for each image
        # h = H/d, w = W/d (height and width of each patch)
        # z_BCHW -> z_BPChw -> valid_z_BPLC -> channel -> padded_z_BPLC_hat -> z_BPChw_hat -> z_BCHW_hat
        decision = []
        
        B, C, H, W = x.size()
        d = self.d
        attention_mask = get_weighted_total_attention_mask(ROI_Index,B,d,d).to(self.device)
        patch_wise_rate = patch_rate_allocation(attention_mask, self.channel_rate_list)
        
        
        d = int(np.sqrt(patch_wise_rate.size()[1]))

        
        if self.training:
            z_BCHW, mask = self.Encoder(x,attention_mask)
            decision.extend(mask)
        else:
            z_BCHW = self.Encoder(x,attention_mask)    
        
        B, C, H, W = z_BCHW.size()
        h = H//d
        w = W//d
            
        z_BPChw = patch_division(z_BCHW,d)
        
        z_BPLC, valid_z_BPLC, mask_BPLC = get_patchwise_valid_z_mask(z_BPChw, patch_wise_rate)
        
        mask_BPCHW = BPLC_to_BPCHW(mask_BPLC,h,w)
        
        mask_BCHW = reverse_patch_division(mask_BPCHW)
        
        valid_z_BCHW = get_valid_z(z_BCHW,mask_BCHW)
               
        
        z_BCHW_hat = self.channel(valid_z_BCHW,SNR_info)
        
        padded_z_BCHW_hat = padding_received_z_hat(z_BCHW, z_BCHW_hat,mask_BCHW)
        
        
        if self.training:
            task_output, mask = self.Decoder(padded_z_BCHW_hat,attention_mask)
            decision.extend(mask)
            return task_output, decision
        else:
            task_output = self.Decoder(padded_z_BCHW_hat,attention_mask)
            return task_output

    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch




class ROIJSCCwoRB(ROIJSCC): # Feature Importance Guide JSCC
    def __init__(self, model_info):
        super(ROIJSCCwoRB, self).__init__(model_info)
        #self.epoch = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        
        # we assume total image is divided with 16 patches
        RONI_C = self.C
        ROI_C = self.C
        self.channel_rate_list = [RONI_C,self.C,ROI_C]
        
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = ROIEncoder(model_info,ROI_C)
        self.channel = Channel(self.chan_type)
        self.Decoder = ROIDecoder(model_info,ROI_C)
        
        self.d = 4
        





