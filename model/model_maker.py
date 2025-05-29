from .common_component import *
from .JSCC import *

def get_model_info(cfg):
    model_info = dict()
    model_info['chan_type'] = cfg.chan_type
    model_info['color_channel'] = cfg.data_info.color_channel
    model_info['rcpp'] = cfg.rcpp #reverse of channel per pixel
    model_info['window_size'] = 8
    model_info['ratio'] = 0.5
    cfg.ratio = model_info['ratio'] #importance ratio
    model_info['gamma'] = 0.5
    cfg.gamma = model_info['gamma']
    model_info['ratio1'] = 0.5
    cfg.ratio1 = model_info['ratio1'] # encoder's importance ratio
    model_info['ratio2'] = 0.5
    cfg.ratio2 = model_info['ratio2'] # decoder's importance ratio
    model_info['eta'] = 0.1

    model_info['window_size_list'] = [8,8,8,8]
    model_info['num_heads_list'] = [4, 6, 8, 10] ##careful! n_feats_list[i]/num_heads_list[i] should be integer
    model_info['input_resolution'] = cfg.input_resolution
    model_info['n_block_list'] = [2,2,2,2]
    
    if cfg.model_name == "ConvJSCC":
        model_info['n_feats_list'] = [32,32,32,32]  
    elif cfg.model_name == "ResJSCC":
        model_info['n_feats_list'] = [32,32,32,32]
    elif cfg.model_name == "SwinJSCC":
        model_info['n_feats_list'] = [40,60,80,160] 
    elif cfg.model_name == "FAJSCC" or cfg.model_name == "ROIJSCC":
        model_info['n_feats_list'] = [40,60,80,260] 
    elif cfg.model_name == "FAJSCCwRL" or cfg.model_name == "FAJSCCwRB" or cfg.model_name == "FAJSCCwRLB":
        model_info['n_feats_list'] = [40,60,80,260] 
    elif cfg.model_name == "ROIJSCCwoRB":
        model_info['n_feats_list'] = [40,60,80,260] 

    else:
        raise ValueError(f'n_feats_list for {cfg.model_name} model is not implemented yet')

    return model_info


def ModelMaker(cfg):
    model = None
    model_info = dict()
    
    model_info = get_model_info(cfg)

    if cfg.model_name == "ConvJSCC":
        model = ConvJSCC(model_info)
    elif cfg.model_name == "ResJSCC":
        model = ResJSCC(model_info)
    elif cfg.model_name == "SwinJSCC":
        model = SwinJSCC(model_info)
    elif cfg.model_name == "FAJSCC":
        model = FAJSCC(model_info)
    elif cfg.model_name == "ROIJSCC":
        model = ROIJSCC(model_info)
       
    elif cfg.model_name == "FAJSCCwRL": #with ROI focusing loss
        model = FAJSCC(model_info)
    elif cfg.model_name == "FAJSCCwRB": #with ROI focusing bandwidth
        model = FAJSCCwRB(model_info)
    elif cfg.model_name == "FAJSCCwRLB": #with ROI focusing loss + bandwidth
        model = FAJSCCwRB(model_info)        
    elif cfg.model_name == "ROIJSCCwoRB":
        model = ROIJSCCwoRB(model_info)    
    
    else:
        raise ValueError(f'{cfg.model_name} model is not implemented yet')
    return model
    
    
    
    
    
    
    
    




















































