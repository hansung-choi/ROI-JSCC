from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
import random
import os



@hydra.main(version_base = '1.1',config_path="configs",config_name='train')
def main(cfg: DictConfig):        
    logger = logging.getLogger(__name__)
    
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
    data_info = DataMaker(cfg)

    # make model
    model = ModelMaker(cfg)   # make model


    # make criterion
    model.d = 4
    d = model.d
    criterion = LossMaker(cfg,d)

    saved_model_epoch = 0

    if cfg.data_info.data_name == "ImageNet" or cfg.data_info.data_name == "DIV2K" or cfg.data_info.data_name == "OpenImage" or True:
        task = cfg.task_name
        data = cfg.data_info.data_name
        chan_type = cfg.chan_type
        SNR = str(cfg.SNR_info).zfill(3)
        rcpp = str(cfg.rcpp).zfill(3)
        random_seed_num = cfg.random_seed
        metric = cfg.performance_metric
        random_num = str(random_seed_num).zfill(3)
        model_name = cfg.model_name
        save_dir = "../../saved_models/" #"../../../saved_models/"
        
        save_name = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}.pt"
        save_name_backup = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"            
          
        model_info_save_path = save_dir + save_name
        model_backup_info_save_path = save_dir + save_name_backup
        if os.path.exists(model_info_save_path):
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
                try:
                    model.load_state_dict(torch.load(model_backup_info_save_path))
                    logger.info(f'The saved backup model is loaded')
                    saved_model_epoch = model.epoch.item()
                    logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
                    random_seed_num = int(saved_model_epoch)
                    torch.manual_seed(random_seed_num)
                    np.random.seed(random_seed_num)
                    random.seed(random_seed_num)
                except Exception as ex:
                    logger.info(f'Error occured during backup model is loaded')
                    logger.info(f'Error info:',ex)

                    logger.info(f'Train epoch is initialized, new default model is made')
                    model = ModelMaker(cfg)   # make model and set appropriate task name
                    

    if saved_model_epoch >=cfg.total_max_epoch:
        logger.info(f'saved model already exists, total_max_epoch is {cfg.total_max_epoch}')
        return None
        
        
    # make optimizer
    optimizer = OptimizerMaker(model,cfg) 

    # make scheduler
    scheduler = None

    
    # make criterion
    model.d = 4
    d = model.d
    criterion = LossMaker(cfg,d)
    get_task_info(cfg)
    
    
    
    logger.info(hydra_cfg['runtime']['output_dir'])
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'Task: {cfg.task_name}')
    logger.info(f'Data: {cfg.data_info.data_name}')
    logger.info(f'chan_type: {cfg.chan_type}')
    logger.info(f'SNR: {cfg.SNR_info}')
    logger.info(f'rcpp: {cfg.rcpp}')
    logger.info(f'performance_metric: {cfg.performance_metric}')
    logger.info(f'Model: {cfg.model_name}')
    logger.info(f'Learning rate: {cfg.learning_rate}')



    # train model
    train_model(cfg, logger, model, data_info.trainloader, data_info.testloader, criterion, optimizer, scheduler)

    task = cfg.task_name
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_num = str(random_seed_num).zfill(3)
    model_name = cfg.model_name
    save_dir = "../../saved_models/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
 







if __name__ == '__main__':
    main()
    
    
    
    
    