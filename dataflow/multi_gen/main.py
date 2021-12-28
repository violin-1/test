from unicodedata import category
import numpy as np
# import matplotlib.pyplot as plt
import os
# import sys
# import torch
import torchvision.transforms as transforms
import cv2
# sys.path.append(os.path.abspath("."))
# import pclpyd as pcl
import random
import os
import torch.optim as optim
# from visualizer.OpencvVis import CVis
# from model import ModelContorller as modelctr

from dataflow.multi_gen.utils import *
from dataflow.datautils import Trainer
from dataflow.multi_gen.vis import set_vis_args
# from utils import *

def args_set(datactr,args):
    datactr.config.Set_datactr_func=None
    if datactr.config.ctrtype == 'train':
        datactr.config.Set_exec_func = Trainer
        datactr.train_func = args['train_func']
        datactr.val_func = args['val_func']
        
    elif datactr.config.ctrtype == 'test':
        datactr.config.Set_exec_func = Trainer
        datactr.train_func = None
        datactr.val_func = args['val_func']
    datactr.set()
    
    datactr.train_init = train_init
    datactr.end_func = end_func
    datactr.res_func = res_compare
    datactr.optimizer = optim.Adam(datactr.modelctr.models[datactr.curmodel].to(datactr.device).parameters(), lr=datactr.config.lr, weight_decay=0.00005)
    datactr.scheduler = optim.lr_scheduler.StepLR(datactr.optimizer, step_size=datactr.config.step_size, gamma=datactr.config.gamma)
    
    datactr.datasets[0].out=args['out']
    datactr.datasets[1].out=args['out']
    # datactr.log = logger(datactr.config.arch,datactr.config.datatsetstype[datactr.curdataset],args['out'],datactr.config.trainratio)
    
def set_multi_datactr(datactr):
    args = {}
    args['train_func'] = train_epoch
    args['val_func'] = val_epoch
    args['out'] = 'multi'
    args_set(datactr,args)
    return

def set_draw_datactr(datactr):
    datactr.config.Set_datactr_func=None
    datactr.exec_func = drawGraph
    return

# def set_vis_arg(datactr):
#     # datactr.vis.set(xyzrelist=[1,-1,1])
#     set_vis_args(datactr)
    
def get_dataset(config):
    datadir = os.path.join('./datasets','multi_modal_Polarization')
    return Multi_modal_polarization(datadir,config.ctrtype)

class Multi_modal_polarization:
    def __init__(self,root,dtype):
        self.dir=None
        self.npoints = None
        self.image_size_w = 256
        self.image_size_h = 256
        self.set(root,dtype)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(self.image_size_w),
            transforms.RandomHorizontalFlip(),
        ])
        pass
    
    def set(self,root=None,dtype=None):
        if root is not None:
            self.dir = root
        if self.dir is not None and dtype is not None:
            self.dolpfiletxt = os.path.join(self.dir,'gen_DOLP_'+dtype+'.txt')
            self.s0filetxt = os.path.join(self.dir,'gen_S0_'+dtype+'.txt')
            self.dtype = dtype
            
            dolpmat = np.genfromtxt(self.dolpfiletxt,delimiter=',', dtype=np.str)
            s0mat = np.genfromtxt(self.s0filetxt,delimiter=',', dtype=np.str)
            
            self.dolpdatapath = dolpmat[:,0].astype(np.str)
            self.s0datapath = s0mat[:,0].astype(np.str)
            
            self.label = dolpmat[:,1].astype(np.int)
        
            self.len = len(self.dolpdatapath)
            self.get_category()
    
    def __len__(self):
        return self.len
        pass
    
    def __getitem__(self,idx):
        dolppath = self.dolpdatapath[idx]
        s0path = self.s0datapath[idx]
        label = self.label[idx]
        category = self.category[idx]
        
        if self.out == 'multi':
            dolp = cv2.resize(cv2.imread(dolppath),(self.image_size_w,self.image_size_h),0,0,cv2.INTER_LINEAR)
            dolp = self.transform(dolp)
            s0 = cv2.resize(cv2.imread(s0path),(self.image_size_w,self.image_size_h),0,0,cv2.INTER_LINEAR)
            s0 = self.transform(s0)
            
            if self.dtype == 'train':
                return s0,dolp,label
            elif self.dtype == 'test':
                return s0,dolp,label,category
        elif self.out == 'Polar':
            dolp = cv2.resize(cv2.imread(dolppath),(self.image_size_w,self.image_size_h),0,0,cv2.INTER_LINEAR)
            dolp = self.transform(dolp)
            if self.dtype == 'train':
                return dolp,label
            elif self.dtype == 'test':
                return dolp,label,category
        elif self.out == 'RGB':
            s0 = cv2.resize(cv2.imread(s0path),(self.image_size_w,self.image_size_h),0,0,cv2.INTER_LINEAR)
            s0 = self.transform(s0)
            if self.dtype == 'train':
                return s0,label
            elif self.dtype == 'test':
                return s0,label,category
            
    def get_category(self):
        self.cat_name = {"面具":0,"A4纸":1,"定制假头":2,"显示屏":3,"相纸":4,"真人":5}
        self.category = []
        for path in self.s0datapath:
            if "dataset1_live" in path:
                self.category.append(self.cat_name["真人"])
                pass
            elif "dataset2_all" in path:
                for name in self.cat_name.keys():
                    if name in path:
                        self.category.append(self.cat_name[name])
                        break
                pass
            elif "dataset3_attack" in path:
                if '1001_' in path or '10001_' in path or '10002_' in path or '10003_' in path or '10004_' in path or '10005_' in path:
                    self.category.append(self.cat_name["显示屏"])
                elif '20001_' in path or '20002_' in path or '20003_' in path:
                    self.category.append(self.cat_name["相纸"])
                elif '30001_' in path or '30002_' in path or '30003_' in path:
                    self.category.append(self.cat_name["A4纸"])
                elif '40001_' in path or '40002_' in path:
                    self.category.append(self.cat_name["定制假头"])
                elif '50001_' in path:
                    self.category.append(self.cat_name["面具"])
                elif '60001_' in path:
                    self.category.append(self.cat_name["显示屏"])
                pass
        pass

if __name__ == '__main__':
    datadir = os.path.join(os.path.abspath('.'),'datasets','multi_modal_Polarization')
    m = Multi_modal_polarization(datadir,'train')
    m.out='multi'
    m.__getitem__(150)

    pass
