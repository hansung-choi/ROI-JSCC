from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from model_eval import *
from save_result_plot import *
import random
import os
import gc



def get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list):
    since = time.time()
    total_eval_dict = dict()
    
    for model_name in model_name_list:
        for rcpp in rcpp_list:
            for SNR in SNR_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                model_eval_dict = get_model_eval_dict(cfg,logger,model_name,rcpp,SNR)
                total_eval_dict[model_save_name] = model_eval_dict
    
    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'total result dict is made in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

    return total_eval_dict

def get_model_save_name(cfg,model_name,rcpp,SNR):
    data = cfg.data_info.data_name
    cfg.model_name = model_name
    get_loss_info(cfg)
    get_task_info(cfg)
    task = cfg.task_name
    cfg.SNR_info = SNR
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
      
    save_name = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}.pt"

    return save_name    
    
def get_loaded_model(cfg,logger,model_name,rcpp,SNR):
    data = cfg.data_info.data_name
    cfg.model_name = model_name
    get_loss_info(cfg)
    get_task_info(cfg)
    task = cfg.task_name
    cfg.SNR_info = SNR
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
    
    if model_name in ["ROIJPEG2000","JPEG2000"]:
        model = ModelMaker(cfg)
        logger.info(f'{model_name} for cpp: 1/{cfg.rcpp} and SNR:{cfg.SNR_info}')
        return model
    
    
    if cfg.model_name in ["ROIJSCCall","ROIJSCCnone"]:
        model_name = "ROIJSCC"
    
    save_dir = "../../saved_models/"    
    save_name = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}.pt"
    save_name_backup = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"
          
    model_info_save_path = save_dir + save_name
    model_backup_info_save_path = save_dir + save_name_backup
    if not os.path.exists(model_info_save_path):
        logger.info(f'There is no trained model')
        logger.info(f'{save_name} does not exist')
        return None

    model = ModelMaker(cfg)
    logger.info(f'load {save_name}')        
    try:
        model.load_state_dict(torch.load(model_info_save_path))
        logger.info(f'The saved model is loaded')
        saved_model_epoch = model.epoch.item()
        logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
        random_seed_num = int(saved_model_epoch)
        torch.manual_seed(random_seed_num)
        np.random.seed(random_seed_num)
        random.seed(random_seed_num)
    except Exception as ex:
        logger.info(f'Error occured during saved model is loaded')
        logger.info(f'Error info:',ex)
        model.load_state_dict(torch.load(model_backup_info_save_path))
        logger.info(f'The saved backup model is loaded')
        saved_model_epoch = model.epoch.item()
        logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
        random_seed_num = int(saved_model_epoch)
        torch.manual_seed(random_seed_num)
        np.random.seed(random_seed_num)
        random.seed(random_seed_num)
    

    # load and make model    
    #model = ModelMaker(cfg)
    #logger.info(f'load {save_name}')
    #model.load_state_dict(torch.load(model_info_save_path))

    return model  
    
    
def get_specific_model_result_dict(cfg: DictConfig, logger, model, trainloader,testloader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    since = time.time()
    evaluater = ModelEvaluater(cfg)

    evaluation_dictionary = evaluater.one_epoch_eval(cfg, logger, model, trainloader, testloader, criterion)
    #Note that below three values are meaningless for JPEG2000, since we use JPEG2000 in the independent CMD subprocess.
    if cfg.model_name in ["ROIJPEG2000","JPEG2000"]:
        GFlops = 0.1 #dummy info
        evaluation_dictionary['GFlops'] = GFlops
        Mmemory = cal_MB(cfg, logger, model)
        evaluation_dictionary['Mmemory'] = Mmemory
        Mparams = get_n_model_params(cfg, logger, model)
        evaluation_dictionary['Mparams'] = Mparams #number of parameters of the model
    else:    
        GFlops = cal_flops(cfg, logger, model)
        evaluation_dictionary['GFlops'] = GFlops
        Mmemory = cal_MB(cfg, logger, model)
        evaluation_dictionary['Mmemory'] = Mmemory
        Mparams = get_n_model_params(cfg, logger, model)
        evaluation_dictionary['Mparams'] = Mparams #number of parameters of the model

    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'model {cfg.model_name} result dict is made in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

    #Important for stable gpu use    
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return evaluation_dictionary


def get_model_eval_dict(cfg,logger,model_name,rcpp,SNR):
    logger.info(f'')
    logger.info(f'')
    logger.info(f'---------------------------------------------------------------')
    evaluation_dictionary = None
    
    #load model
    model = get_loaded_model(cfg,logger,model_name,rcpp,SNR)
    
    if model:
        # make data_info
        logger.info(f'model evaluation is started')
        data_info = DataMaker(cfg)
           
        # make criterion
        d = cfg.patch_d
        criterion = LossMaker(cfg,d)    

        evaluation_dictionary = get_specific_model_result_dict(cfg, logger, model, data_info.trainloader,data_info.testloader, criterion)

    return evaluation_dictionary















    