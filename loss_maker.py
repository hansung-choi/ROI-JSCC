from model.model_maker import *
from data_maker import *
from torch.nn import functional as F
from utils import *
from einops import rearrange
from torch_msssim import ssim_, ms_ssim_, SSIM_, MS_SSIM_


def get_task_info(cfg):

    if cfg.model_name == "ConvJSCC":
        cfg.task_name = "ImageTransmission"
    elif cfg.model_name == "ResJSCC":
        cfg.task_name = "ImageTransmission"
    elif cfg.model_name == "SwinJSCC":
        cfg.task_name = "ImageTransmission"        
    elif cfg.model_name == "FAJSCC":
        cfg.task_name = "FAIT"        
    elif cfg.model_name == "ROIJSCC":
        cfg.task_name = "ROIIT"
 
        
        
    elif cfg.model_name == "FAJSCCwRL": #with ROI focusing loss
        cfg.task_name = "FAGIT"
    elif cfg.model_name == "FAJSCCwRB": #with ROI focusing bandwidth
        cfg.task_name = "FAIT"
    elif cfg.model_name == "FAJSCCwRLB": #with ROI focusing loss + bandwidth
        cfg.task_name = "FAGIT"
        
    elif cfg.model_name == "ROIJSCCwoRB":
        cfg.task_name = "ROIIT"
          
        
    else:
        raise ValueError(f'task for {cfg.model_name} model is not implemented yet')

def get_loss_info(cfg):
    cfg.loss_name = None
    get_task_info(cfg)
    
    if cfg.task_name == "ImageTransmission":
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "IT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "IT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')   
    elif cfg.task_name == "FAIT" :
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "FAIT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "FAIT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')
    elif cfg.task_name == "ROIIT":
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "ROIIT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "ROIIT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')    
    elif cfg.task_name == "FAGIT":
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "FAGIT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "FAGIT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')                                
    else:
        raise ValueError(f'loss function for {cfg.task_name} task is not implemented yet')       

def LossMaker(cfg,d=4): #cfg: DictConfig
    get_loss_info(cfg)
    if cfg.loss_name == "IT_MSE":
        loss = IT_MSE()
    elif cfg.loss_name == "IT_SSIM":
        loss = IT_SSIM() 
    elif cfg.loss_name == "FAIT_MSE":
        loss = FAIT_MSE()
    elif cfg.loss_name == "FAIT_SSIM":
        loss = FAIT_SSIM()
    elif cfg.loss_name == "ROIIT_MSE":
        loss = ROIIT_MSE()
    elif cfg.loss_name == "ROIIT_SSIM":
        loss = ROIIT_SSIM()
    elif cfg.loss_name == "FAGIT_MSE":
        loss = FAGIT_MSE()
    elif cfg.loss_name == "FAGIT_SSIM":
        loss = FAGIT_SSIM()  
  
    else:
        raise ValueError(f'{cfg.loss_name} is not implemented yet')
    return loss


class IT_MSE(torch.nn.Module):
    def __init__(self):
        super(IT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        mse = self.mse(image_hat, image)
        total_loss = mse
        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

        return total_loss, psnr
        
    def get_performance_metric(self):
        return "PSNR"

        
class IT_SSIM(torch.nn.Module):
    def __init__(self):
        super(IT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        ssim = ssim_(image_hat, image, data_range=1, size_average=True)
        total_loss = 1-ssim

        return total_loss, ssim.clone().detach().cpu()
        
    def get_performance_metric(self):
        return "SSIM"



class FAIT_MSE(torch.nn.Module):
    def __init__(self, CA_ratio=0.5,gamma = 0.5):
        super(FAIT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss()
        self.CA_ratio = CA_ratio
        self.gamma = gamma

    def forward(self, image_hat, image,decision=None):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        mse = self.mse(image_hat, image)
        total_loss = mse
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss
        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))
        
        return mse, psnr
        
    def get_performance_metric(self):
        return "PSNR"

        
class FAIT_SSIM(torch.nn.Module):
    def __init__(self, CA_ratio=0.5,gamma = 0.5):
        super(FAIT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.CA_ratio = CA_ratio
        self.gamma = gamma

    def forward(self, image_hat, image,decision=None):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        ssim = ssim_(image_hat, image, data_range=1, size_average=True)
        total_loss = 1-ssim
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss

        return total_loss, ssim.clone().detach().cpu()
        
    def get_performance_metric(self):
        return "SSIM"




class ROIIT_MSE(torch.nn.Module):
    def __init__(self, d=4):
        super(ROIIT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss(reduction='none')
        self.d = int(d)
        self.patch_num = int(d*d)

    def forward(self, image_hat, image, ROI_Index):
        B,C,H,W = image.shape
        
        d = self.d
        ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,d,d)
        peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,d,d)
        RONI_attention_mask = get_RONI_attention_mask(ROI_Index,B,d,d)
        
        ROI_attention_mask = rearrange(ROI_attention_mask,'b h w -> b (h w)').to(self.device)
        peripheral_attention_mask = rearrange(peripheral_attention_mask,'b h w -> b (h w)').to(self.device)
        RONI_attention_mask = rearrange(RONI_attention_mask,'b h w -> b (h w)').to(self.device)
        
    
        #print("loss start")
        #since = time.time()
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)

      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2
               
        image_patch_hat = patch_division(image_hat,d) #B X C X H X W -> B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
        image_patch = patch_division(image,d)
        
        unreduced_mse = self.mse(image_patch_hat, image_patch)
        patch_wise_mse = unreduced_mse.mean(dim=[i for i in range(2,len(image_patch_hat.size()))]).reshape(-1,d*d)
        
        #print("patch_wise_mse[0]:",patch_wise_mse[0])
        
        mse =patch_wise_mse.mean()
        ROI_mse = ROI_attention_mask*patch_wise_mse
        ROI_mse = ROI_mse.mean()*d*d #values from 0 attention values should be excluded during mean operation
        peripheral_mse = peripheral_attention_mask*patch_wise_mse
        peripheral_mse = peripheral_mse.mean()*d*d/peripheral_attention_mask.sum()*B
        RONI_mse = RONI_attention_mask*patch_wise_mse
        RONI_mse = RONI_mse.mean()*d*d/RONI_attention_mask.sum()*B
        
        
        total_loss = mse
        total_loss = total_loss + ROI_mse + 0.5*peripheral_mse
        
        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))
        ROI_psnr = 10 * (np.log(1. / ROI_mse.clone().detach().cpu()) / np.log(10))
        peripheral_psnr = 10 * (np.log(1. / peripheral_mse.clone().detach().cpu()) / np.log(10))
        RONI_psnr = 10 * (np.log(1. / RONI_mse.clone().detach().cpu()) / np.log(10))

        return total_loss, psnr, ROI_psnr, peripheral_psnr, RONI_psnr

        
    def get_performance_metric(self):
        return "PSNR"

        
class ROIIT_SSIM(torch.nn.Module):
    def __init__(self, d=4):
        super(ROIIT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.d = int(d)
        self.patch_num = int(d*d)

    def forward(self, image_hat, image, ROI_Index):
        B,C,H,W = image.shape
        
        d = self.d
        ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,d,d)
        peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,d,d)
        RONI_attention_mask = get_RONI_attention_mask(ROI_Index,B,d,d)
        
        ROI_attention_mask = rearrange(ROI_attention_mask,'b h w -> b (h w)').to(self.device)
        peripheral_attention_mask = rearrange(peripheral_attention_mask,'b h w -> b (h w)').to(self.device)
        RONI_attention_mask = rearrange(RONI_attention_mask,'b h w -> b (h w)').to(self.device)
        
    
        #print("loss start")
        #since = time.time()
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)

      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        full_ssim = ssim_(image_hat, image, data_range=1, size_average=True)
               
        image_patch_hat = patch_division(image_hat,d) #B X C X H X W -> B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
        image_patch = patch_division(image,d)

        image_patch_hat = rearrange(image_hat,'b c (h dh) (w dw) -> (b h w) c dh dw', b=B,c=C,h=d,w=d)   
        image_patch = rearrange(image,'b c (h dh) (w dw) -> (b h w) c dh dw', b=B,c=C,h=d,w=d)   

        unreduced_ssim = ssim_(image_patch_hat, image_patch, data_range=1, size_average=False)
        unreduced_ssim = unreduced_ssim.reshape(-1)
        patch_wise_ssim = rearrange(unreduced_ssim,'(b h w) -> b (h w)', b=B,h=d,w=d)   





        ROI_ssim = ROI_attention_mask*patch_wise_ssim
        ROI_ssim = ROI_ssim.mean()*d*d #values from 0 attention values should be excluded during mean operation
        peripheral_ssim = peripheral_attention_mask*patch_wise_ssim
        peripheral_ssim = peripheral_ssim.mean()*d*d/peripheral_attention_mask.sum()*B
        RONI_ssim = RONI_attention_mask*patch_wise_ssim
        RONI_ssim = RONI_ssim.mean()*d*d/RONI_attention_mask.sum()*B
        
        
        total_loss = 1-full_ssim
        total_loss = total_loss + (1-ROI_ssim) + 0.5*(1-peripheral_ssim)
        
        full_ssim_ = full_ssim.clone().detach().cpu()
        ROI_ssim_ = ROI_ssim.clone().detach().cpu()
        peripheral_ssim_ = peripheral_ssim.clone().detach().cpu()
        RONI_ssim_ = RONI_ssim.clone().detach().cpu()

        return total_loss, full_ssim_, ROI_ssim_, peripheral_ssim_, RONI_ssim_

    def get_performance_metric(self):
        return "SSIM"






class FAGIT_MSE(torch.nn.Module):
    def __init__(self, d=4, CA_ratio=0.5,gamma = 0.5):
        super(FAGIT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss(reduction='none')
        self.d = int(d)
        self.patch_num = int(d*d)
        self.CA_ratio = CA_ratio
        self.gamma = gamma


    def forward(self, image_hat, image, ROI_Index,decision=None):
        B,C,H,W = image.shape
        
        d = self.d
        ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,d,d)
        peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,d,d)
        RONI_attention_mask = get_RONI_attention_mask(ROI_Index,B,d,d)
        
        ROI_attention_mask = rearrange(ROI_attention_mask,'b h w -> b (h w)').to(self.device)
        peripheral_attention_mask = rearrange(peripheral_attention_mask,'b h w -> b (h w)').to(self.device)
        RONI_attention_mask = rearrange(RONI_attention_mask,'b h w -> b (h w)').to(self.device)
        
    
        #print("loss start")
        #since = time.time()
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)

      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2
               
        image_patch_hat = patch_division(image_hat,d) #B X C X H X W -> B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
        image_patch = patch_division(image,d)
        
        unreduced_mse = self.mse(image_patch_hat, image_patch)
        patch_wise_mse = unreduced_mse.mean(dim=[i for i in range(2,len(image_patch_hat.size()))]).reshape(-1,d*d)
        
        #print("patch_wise_mse[0]:",patch_wise_mse[0])
        
        mse =patch_wise_mse.mean()
        ROI_mse = ROI_attention_mask*patch_wise_mse
        ROI_mse = ROI_mse.mean()*d*d #values from 0 attention values should be excluded during mean operation
        peripheral_mse = peripheral_attention_mask*patch_wise_mse
        peripheral_mse = peripheral_mse.mean()*d*d/peripheral_attention_mask.sum()*B
        RONI_mse = RONI_attention_mask*patch_wise_mse
        RONI_mse = RONI_mse.mean()*d*d/RONI_attention_mask.sum()*B
        
        
        total_loss = mse
        total_loss = total_loss + ROI_mse + 0.5*peripheral_mse
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss

        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))
        ROI_psnr = 10 * (np.log(1. / ROI_mse.clone().detach().cpu()) / np.log(10))
        peripheral_psnr = 10 * (np.log(1. / peripheral_mse.clone().detach().cpu()) / np.log(10))
        RONI_psnr = 10 * (np.log(1. / RONI_mse.clone().detach().cpu()) / np.log(10))

        return total_loss, psnr, ROI_psnr, peripheral_psnr, RONI_psnr

        
    def get_performance_metric(self):
        return "PSNR"

        
class FAGIT_SSIM(torch.nn.Module):
    def __init__(self, d=4, CA_ratio=0.5,gamma = 0.5):
        super(FAGIT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.d = int(d)
        self.patch_num = int(d*d)
        self.CA_ratio = CA_ratio
        self.gamma = gamma


    def forward(self, image_hat, image, ROI_Index):
        B,C,H,W = image.shape
        
        d = self.d
        ROI_attention_mask = get_ROI_attention_mask(ROI_Index,B,d,d)
        peripheral_attention_mask = get_peripheral_attention_mask(ROI_Index,B,d,d)
        RONI_attention_mask = get_RONI_attention_mask(ROI_Index,B,d,d)
        
        ROI_attention_mask = rearrange(ROI_attention_mask,'b h w -> b (h w)').to(self.device)
        peripheral_attention_mask = rearrange(peripheral_attention_mask,'b h w -> b (h w)').to(self.device)
        RONI_attention_mask = rearrange(RONI_attention_mask,'b h w -> b (h w)').to(self.device)
        
    
        #print("loss start")
        #since = time.time()
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)

      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        full_ssim = ssim_(image_hat, image, data_range=1, size_average=True)
               
        image_patch_hat = patch_division(image_hat,d) #B X C X H X W -> B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
        image_patch = patch_division(image,d)

        image_patch_hat = rearrange(image_hat,'b c (h dh) (w dw) -> (b h w) c dh dw', b=B,c=C,h=d,w=d)   
        image_patch = rearrange(image,'b c (h dh) (w dw) -> (b h w) c dh dw', b=B,c=C,h=d,w=d)   

        unreduced_ssim = ssim_(image_patch_hat, image_patch, data_range=1, size_average=False)
        unreduced_ssim = unreduced_ssim.reshape(-1)
        patch_wise_ssim = rearrange(unreduced_ssim,'(b h w) -> b (h w)', b=B,h=d,w=d)   





        ROI_ssim = ROI_attention_mask*patch_wise_ssim
        ROI_ssim = ROI_ssim.mean()*d*d #values from 0 attention values should be excluded during mean operation
        peripheral_ssim = peripheral_attention_mask*patch_wise_ssim
        peripheral_ssim = peripheral_ssim.mean()*d*d/peripheral_attention_mask.sum()*B
        RONI_ssim = RONI_attention_mask*patch_wise_ssim
        RONI_ssim = RONI_ssim.mean()*d*d/RONI_attention_mask.sum()*B
        
        
        total_loss = 1-full_ssim
        total_loss = total_loss + (1-ROI_ssim) + 0.5*(1-peripheral_ssim)
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss
        
        full_ssim_ = full_ssim.clone().detach().cpu()
        ROI_ssim_ = ROI_ssim.clone().detach().cpu()
        peripheral_ssim_ = peripheral_ssim.clone().detach().cpu()
        RONI_ssim_ = RONI_ssim.clone().detach().cpu()

        return total_loss, full_ssim_, ROI_ssim_, peripheral_ssim_, RONI_ssim_

    def get_performance_metric(self):
        return "SSIM"








class imagewisePSNR(torch.nn.Module):
    def __init__(self):
        super(imagewisePSNR, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        unreduced_mse = self.mse(image_hat, image)
        image_wise_mse = unreduced_mse.mean(dim=[i for i in range(1,len(image.size()))]).reshape(-1).clone().detach().cpu()

        image_wise_psnr = 10 * (np.log(1. / image_wise_mse) / np.log(10))
        return image_wise_psnr



class imagewiseSSIM(torch.nn.Module):
    def __init__(self):
        super(imagewiseSSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        unreduced_SSIM = ssim_(image_hat, image, data_range=1, size_average=False)
        image_wise_SSIM = unreduced_SSIM.reshape(-1).clone().detach().cpu()

        return image_wise_SSIM







        