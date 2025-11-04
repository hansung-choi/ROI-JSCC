from data_maker import *
import time
import math
import pickle
import matplotlib.pyplot as plt
from utils import *
import gc
import psutil
import os, errno
import random

def save_loss_plot(total_loss_info_list,cfg, logger):

    # get plot info
    epoch_list = [1+i for i in range(len(total_loss_info_list[1]))]

    #plt.rcParams.update({'text.usetex': True})
    #plt.rcParams["figure.figsize"] = (14,8)
    fig, ax1 = plt.subplots()

    lines = ax1.plot(epoch_list, total_loss_info_list[1],label="train loss")

    plt.title(f"loss per epoch")
    ax1.set_xlabel(r'Epochs', fontsize=18)
    ax1.set_ylabel(f'{total_loss_info_list[0]}', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    
    
    # save results
    save_dir = "../../loss_curve_info/" #"../../../loss_curve_info/"
    task = cfg.task_name
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
    model_name = cfg.model_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        

    save_name = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}"    
    
    plot_save_name = save_name + ".pdf"
    if save_name:
        logger.info(f'plot of loss info is saved')
        plt.savefig(save_dir+plot_save_name)
    # plt.show()
    plt.clf()
    
    list_save_name = save_name + ".pkl"
    with open(save_dir+list_save_name,"wb") as f:
        logger.info(f'list of loss info is saved')
        pickle.dump(total_loss_info_list, f)
    
    #with open("save_dir+list_save_name","rb") as f:
        #total_loss_info_list = pickle.load(f)
    
    

def train_model(cfg: DictConfig, logger, model, trainloader, testloader, criterion, optimizer, scheduler=None):
    num_epoch = cfg.train_epoch
    total_max_epoch = cfg.total_max_epoch
    saved_model_epoch = int(model.epoch.item())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    since = time.time()
    trainer = Trainer(cfg)
    
    # save information for loss curve
    total_loss_info_list = [cfg.loss_name]
    train_loss_list = []
    test_loss_list = []
    
    # save inforamtion for best performance model
    best_train_loss = float('inf')    
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
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}.pt"    
    save_name_backup = f"{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"
            
    save_point = 0


    criterion.current_epoch = model.epoch.item()
        
    #logger.info(f'model save epoch point: {save_point+1}')
    
    for epoch in range(saved_model_epoch,total_max_epoch):
        logger.info(f'---------------------------------------------------------------')
        logger.info(f'Epoch {epoch + 1}/{total_max_epoch}')                
        saved_model_epoch = model.epoch.item()
        logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
        random_seed_num = int(saved_model_epoch)
        torch.manual_seed(random_seed_num)
        np.random.seed(random_seed_num)
        random.seed(random_seed_num)

        train_epoch_loss = trainer.one_epoch_train(cfg, logger, model, trainloader, testloader, criterion, optimizer, scheduler)
        
        model.add_epoch()        
        
        train_loss_list.append(train_epoch_loss)
        since1 = time.time()
        if (epoch+1)%1==0 and (epoch+1) >= save_point: #10 #best_train_loss > train_epoch_loss and (epoch+1)%1==0 and epoch > 20
            #best_train_loss = train_epoch_loss
            #torch.save(model.state_dict(), save_dir + save_name)
            #saved_model_epoch = model.epoch.item()
            #logger.info(f'saved_model_total_epoch: {saved_model_epoch}')
            #logger.info(f'The model is saved at train epoch {epoch+1}')
            #torch.save(model.state_dict(), save_dir + save_name_backup)            
            #logger.info(f'backup model is saved at train epoch {epoch+1}')            
            #logger.info(f'One epoch train is finished')
            #return None
            a = 1 #dummy code
            try:
                best_train_loss = train_epoch_loss
                torch.save(model.state_dict(), save_dir + save_name)
                saved_model_epoch = model.epoch.item()
                logger.info(f'saved_model_total_epoch: {saved_model_epoch}')
                logger.info(f'The model is saved at train epoch {epoch+1}')
                if (epoch+1)%1==0 and (epoch+1) >= save_point:
                    torch.save(model.state_dict(), save_dir + save_name_backup)            
                    logger.info(f'backup model is saved at train epoch {epoch+1}')            
                    logger.info(f'One epoch train is finished')
                #return None
            except Exception as ex:
                logger.info(f'Error occured during model save')
                logger.info(f'Error info:',ex)
                try:
                    os.remove(save_dir + save_name)
                except OSError:
                    pass                        
                best_train_loss = train_epoch_loss
                torch.save(model.state_dict(), save_dir + save_name)
                saved_model_epoch = model.epoch.item()
                logger.info(f'saved_model_total_epoch: {saved_model_epoch}')
                logger.info(f'The model is saved at train epoch {epoch+1}')
                try:                        
                    torch.save(model.state_dict(), save_dir + save_name_backup)            
                    logger.info(f'backup model is saved at train epoch {epoch+1}')            
                    logger.info(f'One epoch train is finished')
                except:
                    try:
                        os.remove(save_dir + save_name_backup)
                    except OSError:
                        pass
                    torch.save(model.state_dict(), save_dir + save_name_backup)            
                    logger.info(f'backup model is saved at train epoch {epoch+1}')            
                    logger.info(f'One epoch train is finished')
                #return None
            time_elapsed = time.time() - since1
            logger.info(f'Model save complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')
            

           
        
    total_loss_info_list.append(train_loss_list)
    save_loss_plot(total_loss_info_list,cfg, logger)
    
    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

    return None


class Trainer():

    def __init__(self,cfg: DictConfig):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.task = cfg.task_name

    def one_epoch_train(self,cfg, logger, model, trainloader, testloader, criterion, optimizer,scheduler):
        train_epoch_loss = 0
        if self.task == "ImageTransmission":
            train_epoch_loss = self.train_IT_task(cfg, logger, model, trainloader, testloader, criterion,
                                                              optimizer,scheduler)
        elif self.task == "FAIT":
            train_epoch_loss = self.train_FAIT_task(cfg, logger, model, trainloader, testloader, criterion,
                                                              optimizer,scheduler)
        elif self.task == "ROIIT":
            train_epoch_loss = self.train_ROIIT_task(cfg, logger, model, trainloader, testloader, criterion,optimizer,scheduler)
        
        elif self.task == "FAGIT":
            train_epoch_loss = self.train_FAGIT_task(cfg, logger, model, trainloader, testloader, criterion,optimizer,scheduler)

        else:
            raise ValueError(f'{self.task} task train is not implemented yet')
        return train_epoch_loss
    

        
    def train_IT_task(self,cfg: DictConfig, logger, model, trainloader, testloader, criterion, optimizer, scheduler):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        since = time.time()

        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        
        count = 0
                
        
        for images, labels in trainloader:

            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()

            images_hat = model(images, SNR_info=cfg.SNR_info)
            total_loss = 0.

            total_loss, performance = criterion(images_hat, images)

            total_loss.backward()
            optimizer.step()

            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)


        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_total_loss)
            else:
                scheduler.step()
                
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

        return train_epoch_total_loss        

    def train_FAIT_task(self,cfg: DictConfig, logger, model, trainloader, testloader, criterion, optimizer, scheduler):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        since = time.time()

        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric

        
        count = 0
                
        
        for images, labels in trainloader:

            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()

            images_hat, decision = model(images, SNR_info=cfg.SNR_info)
            total_loss = 0.

            total_loss, performance = criterion(images_hat, images,decision)

            total_loss.backward()
            optimizer.step()

            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)


        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_total_loss)
            else:
                scheduler.step()
                
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

        return train_epoch_total_loss        

    def train_ROIIT_task(self,cfg: DictConfig, logger, model, trainloader, testloader, criterion, optimizer, scheduler):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        since = time.time()

        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        train_epoch_ROI_performance = 0
        train_epoch_ROP_performance = 0 #Region of peripheral
        train_epoch_RONI_performance = 0
        
        performance_metric = cfg.performance_metric
        ROI_performance_metric = "ROI_"+ performance_metric
        ROP_performance_metric = "ROP_"+ performance_metric
        RONI_performance_metric = "RONI_"+ performance_metric
        
        count = 0
                
        
        for images, labels in trainloader:

            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            ROI_Index = (random.randint(0, 3),random.randint(0, 3))
            ROI_Index = (random.randint(1, 2),random.randint(1, 2))
            d = cfg.patch_d
            ROI_Index = (random.randint(1, d-2),random.randint(1, d-2))
            images_hat = model(images, ROI_Index =ROI_Index, SNR_info=cfg.SNR_info)
            total_loss = 0.

            total_loss, performance, ROI_performance, ROP_performance, RONI_performance = criterion(images_hat, images, ROI_Index)

            total_loss.backward()
            optimizer.step()

            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            train_epoch_ROI_performance += ROI_performance.item() * images.size(0)
            train_epoch_ROP_performance += ROP_performance.item() * images.size(0)
            train_epoch_RONI_performance += RONI_performance.item() * images.size(0)


        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_total_loss)
            else:
                scheduler.step()
                
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        train_epoch_ROI_performance = train_epoch_ROI_performance / count
        train_epoch_ROP_performance = train_epoch_ROP_performance / count
        train_epoch_RONI_performance = train_epoch_RONI_performance / count
        
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'{ROI_performance_metric}: {train_epoch_ROI_performance}')
        logger.info(f'{ROP_performance_metric}: {train_epoch_ROP_performance}')
        logger.info(f'{RONI_performance_metric}: {train_epoch_RONI_performance}')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

        return train_epoch_total_loss        


    def train_FAGIT_task(self,cfg: DictConfig, logger, model, trainloader, testloader, criterion, optimizer, scheduler):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        since = time.time()

        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        train_epoch_ROI_performance = 0
        train_epoch_ROP_performance = 0 #Region of peripheral
        train_epoch_RONI_performance = 0
        
        performance_metric = cfg.performance_metric
        ROI_performance_metric = "ROI_"+ performance_metric
        ROP_performance_metric = "ROP_"+ performance_metric
        RONI_performance_metric = "RONI_"+ performance_metric
        
        count = 0
                
        
        for images, labels in trainloader:

            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            ROI_Index = (random.randint(0, 3),random.randint(0, 3))
            ROI_Index = (random.randint(1, 2),random.randint(1, 2))
            images_hat, decision = model(images, ROI_Index =ROI_Index, SNR_info=cfg.SNR_info)
            total_loss = 0.

            total_loss, performance, ROI_performance, ROP_performance, RONI_performance = criterion(images_hat, images, ROI_Index, decision)

            total_loss.backward()
            optimizer.step()

            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            train_epoch_ROI_performance += ROI_performance.item() * images.size(0)
            train_epoch_ROP_performance += ROP_performance.item() * images.size(0)
            train_epoch_RONI_performance += RONI_performance.item() * images.size(0)


        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_total_loss)
            else:
                scheduler.step()
                
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        train_epoch_ROI_performance = train_epoch_ROI_performance / count
        train_epoch_ROP_performance = train_epoch_ROP_performance / count
        train_epoch_RONI_performance = train_epoch_RONI_performance / count
        
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'{ROI_performance_metric}: {train_epoch_ROI_performance}')
        logger.info(f'{ROP_performance_metric}: {train_epoch_ROP_performance}')
        logger.info(f'{RONI_performance_metric}: {train_epoch_RONI_performance}')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

        return train_epoch_total_loss        



