import torchvision
from glob import glob
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
import os, logging
from hydra import initialize, compose
import torch
from hydra.utils import instantiate
import albumentations as A 
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import datasets, transforms
from PIL import Image
import random
import numpy as np
from os.path import join
from os import listdir



def DataMaker(cfg: DictConfig):
    data = None
    if cfg.data_info.data_name == "DIV2K":
        data = DIV2Kdataset(cfg)
    else:
        raise ValueError(f'{cfg.data_info.data_name} is not implemented yet')
    return data


class DIV2Kdataset():
    def __init__(self,cfg: DictConfig):
        self.data_name = cfg.data_info.data_name
        self.batch_size = cfg.data_info.batch_size
        self.num_workers = cfg.data_info.num_workers
        self.num_classes = cfg.data_info.num_classes
        cfg.input_resolution = (256,256)
        print("cfg.input_resolution:",cfg.input_resolution)


        self.train_preprocessor = transforms.Compose([transforms.RandomCrop(256,256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.test_preprocessor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.visual_preprocessor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_dir = '../../../data' #'../../../../data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            
        self.trainset = DatasetFromFolder(data_dir + '/DIV2K/DIV2K_train_HR', transform=self.train_preprocessor)     
        self.testset = Datasets(data_dir + '/DIV2K/DIV2K_valid_HR')
        self.visualset = DatasetFromFolder(data_dir + '/kodak', transform=self.visual_preprocessor)


        self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=self.batch_size,shuffle=True,
                                                       num_workers=self.num_workers,pin_memory=True,drop_last=False)
                                                       
        self.testloader =  torch.utils.data.DataLoader(dataset=self.testset,batch_size=1,shuffle=False)
        
        self.visualloader = torch.utils.data.DataLoader(dataset=self.visualset, batch_size=1,shuffle=False,
                                                       num_workers=1,pin_memory=True,drop_last=False)
        
        if cfg.test_resolution >=100:
            self.test_preprocessor2 = transforms.Compose([transforms.CenterCrop((cfg.test_resolution,cfg.test_resolution)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.testset = DatasetFromFolder(data_dir + '/DIV2K/DIV2K_valid_HR', transform=self.test_preprocessor2) 
            self.testloader =  torch.utils.data.DataLoader(dataset=self.testset,batch_size=1,shuffle=False)

##https://github.com/leftthomas/SRGAN/blob/master/data_utils.py
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.transform = transform


    def __getitem__(self, index):
           
        image = self.transform(Image.open(self.image_filenames[index]))
        label = 0
        return image, label

    def __len__(self):
        return len(self.image_filenames)


##https://github.com/semcomm/SwinJSCC/blob/main/data/datasets.py
class Datasets(Dataset):
    def __init__(self, dataset_dir):
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        


    def __getitem__(self, item):
        image = Image.open(self.image_filenames[item]).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        img = self.transform(image)
        label = 0
        return img, label
    def __len__(self):
        return len(self.image_filenames)













  
