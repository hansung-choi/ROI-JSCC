from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from total_eval import *
import random
import os



@hydra.main(version_base = '1.1',config_path="configs",config_name='model_eval')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    #cfg.patch_d = 4 #This sets the ROI patch setting, default is 4. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'device: {device}')
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # set random seed number
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)
    
    # make data_info
    #cfg.test_resolution = 128 This sets the resolution of test data. If we do not set, we use full resolution test data
    data_info = DataMaker(cfg)        

    model_name_list = ["ROIJSCC","FAJSCC","SwinJSCC","ResJSCC","ConvJSCC"]    


    rcpp_list=[12]
    SNR_list=[1,4,7,10]

    

    total_eval_dict = get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list)

    for rcpp in rcpp_list:
        save_SNR_performance_opt_Avg_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,"ROI")  #Option + Average performance    
    
        save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list)
        save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,"ROI")


        
    
if __name__ == '__main__':
    main()
    














    