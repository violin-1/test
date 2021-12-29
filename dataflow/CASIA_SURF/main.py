from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import math
import cv2
import torchvision.transforms as transforms
import torch

from dataflow.CASIA_SURF.utils import train_epoch,val_epoch,train_init,res_compare,end_func
from dataflow.datautils import Trainer
# from dataflow.CASIA_SURF.log import logger
import torch.optim as optim
# from utils import *

# CASIA-SURF training dataset and our private dataset

def train_set(datactr,args):
       
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
    datactr.res_func = res_compare
    datactr.end_func = end_func
    datactr.optimizer = optim.Adam(datactr.modelctr.models[datactr.curmodel].parameters(), lr=datactr.config.lr, weight_decay=0.00005)
    datactr.scheduler = optim.lr_scheduler.StepLR(datactr.optimizer, step_size=datactr.config.step_size, gamma=datactr.config.gamma)
    datactr.datasets[datactr.curdataset].out=args['out']
    # datactr.log = logger(datactr.config.arch,datactr.config.datatsetstype[datactr.curdataset],args['out'],datactr.config.trainratio)
    

def set_datactr(datactr):
    args = {}
    args['train_func'] = train_epoch
    args['val_func'] = val_epoch
    args['out'] = 'multi'
    train_set(datactr,args)
    return

def set_draw_datactr(datactr):
    datactr.config.Set_datactr_func=None
    datactr.exec_func = None
    return

def set_vis_args(datactr):
    datactr.vis.set(xyzrelist=[1,-1,1])
    
def get_dataset(config):
    datadir = os.path.join('datasets','CASIA_SURF')
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],  ##accorcoding to casia-surf val to commpute
                                     std=[0.10050353, 0.100842826, 0.10034215])
    img_size=256
    return CASIA(
        os.path.join(datadir,'Mytest_train.txt'),
        os.path.join(datadir,'Mytest_test.txt'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]),phase_train=True)
    
class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

class CASIA(Dataset):
    def __init__(self,traindir=None,valdir=None, transform=None, phase_train=True):
        self.npoints = None
        self.phase_train = phase_train
        self.transform = transform
        self.channel = None
        try:
            if traindir is not None:
                with open(traindir, 'r') as f:
                    self.dir_train = f.read().splitlines()
                    self.rgb_dir_train = []
                    self.depth_dir_train = []
                    self.ir_dir_train = []
                    self.label_dir_train = []
                    for line in self.dir_train:
                        self.rgb_dir_train.append(line.split(',')[0])
                        self.depth_dir_train.append(line.split(',')[1])
                        self.ir_dir_train.append(line.split(',')[2])
                        self.label_dir_train.append(int(line.split(',')[3]))
                # with open(label_dir_train_file, 'r') as f:
                #     self.label_dir_train = f.read().splitlines()
            if valdir is not None:
                with open(valdir, 'r') as f:
                    self.dir_val = f.read().splitlines()
                    self.rgb_dir_test = []
                    self.depth_dir_test = []
                    self.ir_dir_test = []
                    self.label_dir_test = []
                    for line in self.dir_val:
                        self.rgb_dir_test.append(line.split(',')[0])
                        self.depth_dir_test.append(line.split(',')[1])
                        self.ir_dir_test.append(line.split(',')[2])
                        self.label_dir_test.append(int(line.split(',')[3]))
            self.cat_name = {}
                # with open(label_dir_val_file, 'r') as f:
                #     self.label_val = f.read().splitlines()
            # if self.phase_test:
            #     with open(depth_dir_test_file, 'r') as f:
            #         self.depth_dir_test = f.read().splitlines()
            #     with open(label_dir_test_file, 'r') as f:
            #         self.label_dir_test = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()
    def set(self,phase_train=True):
        self.phase_train = phase_train
        
    def __len__(self):
        if self.phase_train:
            return len(self.dir_train)
        else:
            return len(self.dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            rgb_dir = self.rgb_dir_train[idx]
            depth_dir = self.depth_dir_train[idx]
            # label_dir = self.label_dir_train[idx]
            label = self.label_dir_train[idx]
            label = np.array(label)
        else:
            rgb_dir = self.rgb_dir_test[idx]
            depth_dir = self.depth_dir_test[idx]
            label = self.label_dir_test[idx]
            # label = int(label_dir[idx])
            label = np.array(label)

        rgb = Image.open(rgb_dir)
        rgb = rgb.convert('RGB')
        
        depth = Image.open(depth_dir)
        depth = depth.convert('RGB')

        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        if self.phase_train:
            return rgb,depth,label
        else:
            return rgb,depth,label,0
