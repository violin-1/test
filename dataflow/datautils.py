import numpy as np
# import matplotlib.pyplot as plt
import sys
import os
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm

def point2_data_pre_deal(pointcloud):
    pointcloud = torch.Tensor(pointcloud)
    return pointcloud.float().permute(1,0).cuda().view(1,pointcloud.shape[1],pointcloud.shape[0])
  
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AvgrageMeter(object):
    
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    
class Trainer:
  def __init__(self,datactr=None):
    if datactr is not None:
        if datactr.train_init is not None:
          datactr.train_init(self,datactr)
        else:
          self.sets(datactr)
        self.train_epoch()
  
  def set(self,datactr=None):
      if datactr is not None:
        self.device = torch.device('cuda:' + str(datactr.config.gpus[0]) if torch.cuda.is_available() else "cpu")
      
      self.lr = datactr.config.lr
      
      self.optimizer = datactr.optimizer
      self.scheduler = datactr.scheduler
      
      if datactr.modelctr.checkpoints[datactr.curmodel] is not None:
          if 'optimizer' in datactr.modelctr.checkpoints[datactr.curmodel].keys():
              self.optimizer.load_state_dict(datactr.modelctr.checkpoints[datactr.curmodel]['optimizer'])
          if 'scheduler' in datactr.modelctr.checkpoints[datactr.curmodel].keys():
              self.scheduler.load_state_dict(datactr.modelctr.checkpoints[datactr.curmodel]['scheduler'])
          self.best_res = datactr.modelctr.checkpoints[datactr.curmodel]['best_res']
          self.strart_epoch = datactr.modelctr.checkpoints[datactr.curmodel]['epoch']
          
      elif os.path.exists(os.path.join(self.savedir,'best.pth')):
          checkpoint = torch.load(os.path.join(self.savedir,'best.pth'))
          datactr.modelctr.models[datactr.curmodel].load_state_dict(checkpoint['state_dict'])
          datactr.modelctr.checkpoints[datactr.curmodel] = checkpoint
          if 'optimizer' in checkpoint.keys():
              self.optimizer.load_state_dict(checkpoint['optimizer'])
          if 'scheduler' in checkpoint.keys():
              self.scheduler.load_state_dict(checkpoint['scheduler'])
          self.best_res = checkpoint['best_res']
          self.strart_epoch = checkpoint['epoch']
      else:
          self.best_acc = 0
          self.strart_epoch = 0
      
      if self.device != 'cpu':
          cudnn.benchmark = True
          # torch.cuda.manual_seed_all(datactr.config.random_seed)
          datactr.config.gpus = [int(i) for i in datactr.config.gpus.split(',')]
          datactr.modelctr.models[datactr.curmodel].to(self.device)
          self.gpus = datactr.config.gpus
          if len(datactr.config.gpus) > 1:
              datactr.modelctr.models[datactr.curmodel] = torch.nn.DataParallel(datactr.modelctr.models[datactr.curmodel],device_ids=datactr.config.gpus)
      
      
      self.end_epoch = datactr.config.epochs
      self.step_size = datactr.config.step_size
      self.gamma = datactr.config.gamma
      self.model = datactr.modelctr.models[datactr.curmodel]
      self.dataset = datactr.datasets[datactr.curdataset]
      self.batch_size = datactr.config.batchsize
      self.num_workers = datactr.config.num_workers
      self.criterion = datactr.modelctr.lossfuc[datactr.curmodel]
      self.resdeal = datactr.modelctr.resdeal[datactr.curmodel]
      
      self.train_func = None
      if datactr.train_func is not None:
        self.train_func = datactr.train_func 
      
      self.val_func = None
      if datactr.val_func is not None:
        self.val_func = datactr.val_func
        
      self.res_func = None
      if datactr.res_func is not None:
        self.res_func = datactr.res_func

      self.end_func = None
      if datactr.end_func is not None:
        self.end_func = datactr.end_func
        
      self.trainratio = datactr.config.trainratio
  
  def train_epoch(self):
    
    for epoch in range(self.strart_epoch,self.end_epoch):
      self.epoch = epoch
      
      if self.train_func is not None:
        if (epoch +1) % self.step_size == 0:
          self.lr *= self.gamma
        self.train_func(self)
        # self.log3(epoch,train_acc,self.loss)
        self.scheduler.step()
        # self.log3(epoch,val_acc,train_acc,self.loss)
      
      if self.val_func is not None:
        self.val_func(self)
        # val_acc = self.acc/self.dataset.__len__()
        
      if self.res_func is not None:
        self.res_func(self)    
      # is_best = val_acc > self.best_acc
      # self.best_acc = max(val_acc, self.best_acc)
      if self.train_func is not None:
        if len(self.gpus) > 1:
          save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': self.model.module.state_dict(),
          'best_res': self.best_res,
          'optimizer': self.optimizer.state_dict(),
          'scheduler': self.scheduler.state_dict(),
          }, filename=os.path.join(self.savedir,'{}.pth'.format(epoch)) )
        else:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': self.model.state_dict(),
              'best_res': self.best_res,
              'optimizer': self.optimizer.state_dict(),
              'scheduler': self.scheduler.state_dict(),
          }, filename=os.path.join(self.savedir,'{}.pth'.format(epoch)) )
        
        if self.is_best :
          if len(self.gpus) > 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'best_res': self.best_res,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, filename=os.path.join(self.savedir,'best.pth') )
          else:
            save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': self.model.state_dict(),
              'best_res': self.best_res,
              'optimizer': self.optimizer.state_dict(),
              'scheduler': self.scheduler.state_dict(),
            }, filename=os.path.join(self.savedir,'best.pth') )
        
        # print('epoch: {} The best is {} last best is {}'.format(epoch,val_acc,self.best_acc))
    if self.end_func is not None:
      self.end_func(self)
    pass