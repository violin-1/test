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

from dataflow.multi_modal_Polarization.utils import *
from dataflow.datautils import Trainer
# from utils import *

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
    datactr.end_func = end_func
    datactr.res_func = res_compare
    datactr.optimizer = optim.Adam(datactr.modelctr.models[datactr.curmodel].parameters(), lr=datactr.config.lr, weight_decay=0.00005)
    datactr.scheduler = optim.lr_scheduler.StepLR(datactr.optimizer, step_size=datactr.config.step_size, gamma=datactr.config.gamma)
    datactr.datasets[datactr.curdataset].out=args['out']
    # datactr.log = logger(datactr.config.arch,datactr.config.datatsetstype[datactr.curdataset],args['out'],datactr.config.trainratio)
    
def set_RGB_datactr(datactr):
    args = {}
    args['train_func'] = single_train_epoch
    args['val_func'] = single_val_epoch
    args['out'] = 'RGB'
    train_set(datactr,args)
    return

def set_Polar_datactr(datactr):
    args = {}
    args['train_func'] = single_train_epoch
    args['val_func'] = single_val_epoch
    args['out'] = 'Polar'
    train_set(datactr,args)
    return

def set_multi_datactr(datactr):
    args = {}
    args['train_func'] = multi_train_epoch
    args['val_func'] = multi_val_epoch
    args['out'] = 'multi'
    train_set(datactr,args)
    return

def set_draw_datactr(datactr):
    datactr.config.Set_datactr_func=None
    datactr.exec_func = drawGraph
    return

def set_vis_arg(datactr):
    datactr.vis.set(xyzrelist=[1,-1,1])
    
def get_dataset(config):
    datadir = os.path.join('./datasets','multi_modal_Polarization')
    return Multi_modal_polarization(datadir,config.ctrtype)

def identify(root):
    if 'live' in root or '真人' in root:
        return True
    return False

def findsamename(root):
    dolpfiles = os.listdir(os.path.join(root,'DOLP'))
    s0files = os.listdir(os.path.join(root,'S0'))
    s0flag = False
    if '.png_s0' in s0files[0]:
        s0flag=True
        _s0files = []
        for s0file in s0files:
            temp = s0file.split('.png_s0')
            _s0files.append(temp[0]+temp[1])
        s0files = _s0files
    
    dolpflag = False
    if '.png_dolp' in dolpfiles[0]:
        dolpflag = True
        _dolpfiles = []
        for dolpfile in dolpfiles:
            temp = dolpfile.split('.png_dolp')
            _dolpfiles.append(temp[0]+temp[1])
        dolpfiles = _dolpfiles
          
    dolpset,s0set = set(dolpfiles),set(s0files)
    subfiles = list(dolpset&s0set)
    
    
    _s0files = []
    for s0file in subfiles:
        if s0flag:
            temp = s0file.split('.')
            name = os.path.join(root,'S0',temp[0]+'.png_s0.'+temp[1])
            _s0files.append((name,int(identify(name))))
        else:
            name = os.path.join(root,'S0',s0file)
            _s0files.append((name,int(identify(name))))
    s0files = np.array(_s0files)
    
    _dolpfiles = []
    for dolpfile in subfiles:
        if dolpflag:
            temp = dolpfile.split('.')
            name = os.path.join(root,'DOLP',temp[0]+'.png_dolp.'+temp[1])
            _dolpfiles.append((name,int(identify(name))))
        else:
            name = os.path.join(root,'DOLP',dolpfile)
            _dolpfiles.append((name,int(identify(name))))
    dolpfiles = np.array(_dolpfiles)
    dolpreal,dolpfake,s0real,s0fake = [],[],[],[]
    ind_ = np.where(dolpfiles[:,1]=='0')[0]
    ind = np.where(dolpfiles[:,1]=='1')[0]
    dolpreal = list(dolpfiles[ind])
    dolpfake = list(dolpfiles[ind_])
    s0real = list(s0files[ind])
    s0fake = list(s0files[ind_])
    
    return dolpreal,dolpfake,s0real,s0fake

def searchalldata(root,flag=False):
    dolpreal,dolpfake,s0real,s0fake = [],[],[],[]
    if flag:
        items = []
        items.append(os.path.join('dataset1_live','HUT','Only_Face'))
        items.append(os.path.join('dataset2_all'))
        items.append(os.path.join('dataset3_attack','HUT','Only_Face'))
    else:
        items = os.listdir(root)
    for item in items:
        # if item.__len__() <=1 or item == 'Deg' or '.txt' in item or 'S2' in item or 'S3' in item:
        #     continue
        if '.' in item:
            if identify(root):
                if 'DOLP' in root:
                    dolpreal.append((os.path.join(root,item),1))
                elif 'S0' in root:
                    s0real.append((os.path.join(root,item),1))
            else:
                if 'DOLP' in root:
                    dolpfake.append((os.path.join(root,item),0))
                elif 'S0' in root:
                    s0fake.append((os.path.join(root,item),0))
        else:
            if 'dataset3_attack' in item or 'HUT' in item:
                tmp1,tmp2,tmp3,tmp4 = findsamename(os.path.join(root,item))
                dolpreal = dolpreal + tmp1
                dolpfake = dolpfake + tmp2
                s0real = s0real + tmp3
                s0fake = s0fake + tmp4
            else:
                tmp1,tmp2,tmp3,tmp4 = searchalldata(os.path.join(root,item))
                dolpreal = dolpreal + tmp1
                dolpfake = dolpfake + tmp2
                s0real = s0real + tmp3
                s0fake = s0fake + tmp4
    return dolpreal,dolpfake,s0real,s0fake

def shuffle_split_data(real,fake,testratio=0.2):
    random.shuffle(real)
    random.shuffle(fake)
    len1 = len(real)
    len2 = len(fake)
    tlen1 = int(len1*testratio)
    tlen2 = int(len2*testratio)
    
    realtest,faketest = real[:tlen1],fake[:tlen2]
    realtrain,faketrain = real[tlen1:],fake[tlen2:]

    return realtrain,faketrain,realtest,faketest

def gendatalist(root,testratio=0.2):
    dolpreal,dolpfake,s0real,s0fake = searchalldata(root,True)
    indreal = np.arange(0,len(dolpreal))
    indfake = np.arange(0,len(dolpfake))
    indrealtrain,indfaketrain,indrealtest,indfaketest = shuffle_split_data(indreal,indfake,testratio)
    dolpreal,dolpfake,s0real,s0fake = np.array(dolpreal),np.array(dolpfake),np.array(s0real),np.array(s0fake)
    dolptraindir,dolptestdir = np.vstack((dolpreal[indrealtrain],dolpfake[indfaketrain])),np.vstack((dolpreal[indrealtest],dolpfake[indfaketest]))
    s0traindir,s0testdir = np.vstack((s0real[indrealtrain],s0fake[indfaketrain])),np.vstack((s0real[indrealtest],s0fake[indfaketest]))
    return dolptraindir,dolptestdir,s0traindir,s0testdir

def save_train_test_to_txt(traindir,testdir,txtsavepath,filenameprefix):
    # traindir,testdir=gendatalist(datadir)
    # with open(os.path.join(txtsavepath,filenameprefix+'all.txt'),'w') as f:
    #     all_ = traindir + testdir
    #     for line in all_:
    #         f.write(line[0]+','+str(line[1])+'\n')
            
    with open(os.path.join(txtsavepath,filenameprefix+'train.txt'),'w') as f:
        for line in traindir:
            f.write(line[0]+','+str(line[1])+'\n')
    
    with open(os.path.join(txtsavepath,filenameprefix+'test.txt'),'w') as f:
        for line in testdir:
            f.write(line[0]+','+str(line[1])+'\n')

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
            self.dolpfiletxt = os.path.join(self.dir,'DOLP_'+dtype+'.txt')
            self.s0filetxt = os.path.join(self.dir,'S0_'+dtype+'.txt')
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
def change(root):
    root = os.path.join(root,'dataset2_all')
    types = os.listdir(root)
    for typ in types:
        if typ == '1' or typ == '2':
            continue;
        else:
            x1 =  os.listdir(os.path.join(root,typ,'DOLP'))
            
            x2 =  os.listdir(os.path.join(root,typ,'S0'))
            if '真人' in typ:
                x1.sort(key= lambda x:int(x[x.find('(')+1:x.find(')')]))
                x2.sort(key= lambda x:int(x[x.find('(')+1:x.find(')')]))
            for i in range(len(x1)-1,-1,-1):
                if x1[i] == x2[i]:
                    continue
                os.rename(os.path.join(root,typ,'S0',x2[i]),os.path.join(root,typ,'S0',x1[i]))
            pass
    
    pass

if __name__ == '__main__':
    datadir = os.path.join(os.path.abspath('.'),'datasets','multi_modal_Polarization')
    mode = 'gen'
    if mode == 'gen':
        dolptraindir,dolptestdir,s0traindir,s0testdir = gendatalist(datadir,0.9)
        save_train_test_to_txt(dolptraindir,dolptestdir,datadir,'DOLP_')
        save_train_test_to_txt(s0traindir,s0testdir,datadir,'S0_')     
    elif mode == 'test':
        m = Multi_modal_polarization(datadir,'train')
        m.out='multi'
        m.__getitem__(150)
    elif mode == 'change':
        change(datadir)
    pass
