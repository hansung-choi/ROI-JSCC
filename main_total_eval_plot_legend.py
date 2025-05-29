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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'device: {device}')
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    model_name_list1 = ["ROI-JSCC (ROI)","FAJSCC (ROI)","SwinJSCC (ROI)","ResJSCC (ROI)","ConvJSCC (ROI)"]
    model_name_list2 = ["ROI-JSCC (Avg)","FAJSCC (Avg)","SwinJSCC (Avg)","ResJSCC (Avg)","ConvJSCC (Avg)"]
    model_name_list = model_name_list1 + model_name_list2
    save_plot_legend_type1(cfg,logger,model_name_list,plot_name='_main_',ncol=5)
    save_plot_legend_type1(cfg,logger,model_name_list,plot_name='_main_',ncol=4)


    model_name_list = ["ROI-JSCC","ROI-JSCC w/o RB","FAJSCC w/ RLB","FAJSCC w/ RB","FAJSCC w/ RL","FAJSCC"]
    save_plot_legend_type2(cfg,logger,model_name_list,plot_name='_ablation_',ncol=3)
    save_plot_legend_type2(cfg,logger,model_name_list,plot_name='_ablation_',ncol=2)

    
    
if __name__ == '__main__':
    main()
    














    