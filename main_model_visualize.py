from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from model_eval import *
import random
import os


@hydra.main(version_base = '1.1',config_path="configs",config_name='model_eval')
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

    
    print("cfg.rcpp:",cfg.rcpp)
    print("cfg.SNR_info:",cfg.SNR_info)
    print("cfg.model_name:",cfg.model_name)
    print("cfg.data_info:",cfg.data_info)
    
    # make data_info
    data_info = DataMaker(cfg)

    # make model
    model = ModelMaker(cfg)   # make model and set appropriate task name

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
    logger.info(f'Random seed: {cfg.random_seed}')
    logger.info(f'Model: {cfg.model_name}')

    task = cfg.task_name
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
    model_name = cfg.model_name
    save_dir = "../../saved_models/" #"../../../saved_models/"

        
    save_name = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}.pt"    
    save_name_backup = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"
          
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

    # visualize task result
    visualize_model(cfg, logger, model, data_info.trainloader,data_info.testloader,data_info.visualloader)











if __name__ == '__main__':
    main()
    
    