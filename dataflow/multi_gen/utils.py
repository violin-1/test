import torch
from torch.utils import data
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from dataflow.multi_gen.log import logger
from dataflow.datautils import AvgrageMeter

def one_hot(batchsize,numclass,tensor):
    return torch.zeros(batchsize,numclass).to(tensor.device).scatter_(1,tensor.view(1,tensor.shape[0]).long().t(),1)

def drawGraph(datactr):
    logfile = datactr.config.draw_logfile
    logpath = os.path.join('model',datactr.config.arch,'logs',logfile)
    best_acc = 0
    valacc = []
    trainloss = []
    epoch = []
    def abstr(line, strfind, strend): 
        return line[line.find(strfind)+len(strfind):line.find(strfind)+len(strfind)+line[line.find(strfind)+len(strfind):].find(strend)]
    
    with open(logpath,'r') as f:
        for line in f.readlines():
            if '||epoch:' in line:
                tmp = abstr(line, '||epoch:', ',')
                epoch.append(int(tmp))
                tmp = abstr(line,'Val_accurate = ',',')
                if(float(tmp) > best_acc):
                    best_acc = float(tmp)
                valacc.append(best_acc)
                tmp = abstr(line,'Train_loss = ',' ')
                trainloss.append(float(tmp))
    valacc = np.array(valacc)*100
    import matplotlib.pyplot as plt
    plt.figure(figsize=[10,5])
    plt.plot(epoch, valacc, label='CDCN_MultiModality')
    plt.title('Val_Acc_Curve best_acc:{}'.format(best_acc))
    plt.xlabel('epoch')
    plt.ylabel('Accurate(%)')
    plt.savefig(os.path.join('outputs','CDCN_MultiModality_Val_Acc.png'))
    
    plt.figure(figsize=[10,5])
    plt.plot(epoch, trainloss, label='CDCN_MultiModality')
    plt.title('Train_Loss_Curve')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('outputs','CDCN_MultiModality_Train_Loss.png'))
    pass

def train_init(Trainer,datactr):
    Trainer.log = logger(datactr.config.arch,datactr.config.datatsetstype[datactr.curdataset],datactr.datasets[datactr.curdataset].out,datactr.config.trainratio)
    Trainer.log.log1()
    Trainer.savedir = Trainer.log.savedir
    Trainer.best_res = None
    Trainer.set(datactr)
    Trainer.args_val = AvgrageMeter()
    Trainer.train_loss = AvgrageMeter()
    Trainer.train_dataset = datactr.datasets[0]
    Trainer.train_dataset.set(dtype = 'train')
    Trainer.dataloader_train = DataLoader(Trainer.train_dataset,batch_size= Trainer.batch_size, shuffle=True,num_workers=Trainer.num_workers)
    Trainer.val_dataset = datactr.datasets[1]
    Trainer.val_dataset.set(dtype = 'test')
    Trainer.dataloader_val = DataLoader(Trainer.val_dataset,batch_size=Trainer.batch_size, shuffle=True,num_workers=Trainer.num_workers)
    
    
def train_epoch(Trainer):
    Trainer.model.train()
    Trainer.train_loss.reset()
    Trainer.log.log2(Trainer.epoch)

    for i,(s0,dolp,_) in enumerate(tqdm(Trainer.dataloader_train)):
        s0,dolp = Variable(s0).float().to(Trainer.device),Variable(dolp).float().to(Trainer.device)
        Trainer.optimizer.zero_grad()
        
        pred =  Trainer.model(s0)

        loss = Trainer.criterion(pred,dolp)
        
        loss.backward()
    
        Trainer.optimizer.step()
        Trainer.train_loss.update(loss.item(),dolp.shape[0])
    
    Trainer.log.log3(Trainer.epoch,Trainer.train_loss.avg)
    pass

def val_epoch(Trainer):
    Trainer.model.eval()
    Trainer.args_val.reset()
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold

        for i,(s0,dolp,_,_) in enumerate(tqdm(Trainer.dataloader_val)):
            # get the inputs
            s0,dolp = Variable(s0).float().to(Trainer.device),Variable(dolp).float().to(Trainer.device)
            Trainer.optimizer.zero_grad()
            pred = Trainer.model(s0)
            
            loss = Trainer.criterion(pred,dolp)
            Trainer.args_val.update(loss.item(),dolp.shape[0])
    Trainer.log.log4(Trainer.epoch,Trainer.args_val.avg)

def res_compare(Trainer):
    Trainer.is_best = False
    if Trainer.best_res  is None or Trainer.args_val.avg < Trainer.best_res:
        Trainer.best_res = Trainer.args_val.avg
        Trainer.is_best = True
    Trainer.log.log_best(Trainer.epoch,Trainer.args_val.avg,Trainer.best_res)

def end_func(Trainer):
    Trainer.log.log5()

