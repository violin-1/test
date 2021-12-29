import numpy as np
import importlib

import time
import os
# from visualizer.OpencvVis import CVis
from model import ModelContorller as modelctr
from dataflow.datautils import *

class DataController:
    def __init__(self,config):
                #  ctrtype='vis',
                #  datatsetstype=['ZedCamera'],
                #  modeltypes=None,
                #  device = 'gpu',
                #  set_datactr_func = None,
                #  set_modelctr_func = None,):
        self.config = None
        self.train_init = None
        self.exec_func = None
        # if config.ctrtype == 'Vis':
        #     self.vis = CVis(grab_pcl_fuc=self.get_data,
        #                 switch_model_func=self.switch_model,
        #                 switch_dataset_func=self.switch_dataset,
        #                 switch_pcl_func=self.switch_pcl,
        #                 )
        self.set(config)
        # self.set(ctrtype,datatsetstype,modeltypes,device,set_datactr_func,set_modelctr_func)
            # self.datasets = []
            # self.set_vis_arg_func = []
            # self.curdataset = 0
            # self.curpcl = 0
            # for i,dataset in enumerate(datatsetstype):
            #     self.datasets.append(importlib.import_module('dataflow.'+dataset+'.main').get_dataset())
            #     self.set_vis_arg_func.append(importlib.import_module('dataflow.'+dataset+'.main').set_vis_arg)
            # self.set_vis_arg_func[self.curdataset](self)
            # self.modelctr = modelctr.ModelController(modeltypes=modeltypes,arg_set_func=set_modelctr_func)
            # self.curmodel = 0

            # self.set_npoints()
            # self.device = device
        pass
    
    def set(self,config=None):
            # ctrtype=None,
            # datatsetstype=None,
            # modeltypes=None,
            # device=None,
            # set_datactr_func=None,
            # set_modelctr_func=None,):
        if config is None and self.config is None:
            return
        elif config is None and self.config is not None:
            config = self.config
        else:
            self.config = config
        if config.Set_datactr_func is not None:
            config.Set_datactr_func(self)
        else:
            if config.datatsetstype is not None and config.ctrtype is not None:
                self.datasets = []
                self.datasets_vis_arg_func = []
                self.curdataset = 0
                self.curpcl = 0
                for i,dataset in enumerate(config.datatsetstype):
                    self.datasets.append(importlib.import_module('dataflow.'+dataset+'.main').get_dataset(config))
                    self.datasets_vis_arg_func.append(importlib.import_module('dataflow.'+dataset+'.main').set_vis_args)
                    
            if config.modeltypes is not None:
                self.modelctr = modelctr.ModelController(modeltypes=config.modeltypes,arg_set_func=config.Set_modelctr_func)
                self.curmodel = 0
                self.device = config.device
            
            if config.Set_exec_func is not None:
                self.exec_func = config.Set_exec_func
            self.set_npoints()
    def get_dataloader(self):
        pass
    
    def set_npoints(self):
        if self.datasets[self.curdataset].npoints is not None and self.modelctr.npoints[self.curmodel] is not None:
            self.npoints = min(self.datasets[self.curdataset].npoints,self.modelctr.npoints[self.curmodel])
        elif self.datasets[self.curdataset].npoints is not None:
            self.npoints = self.datasets[self.curdataset].npoints
        elif self.modelctr.npoints[self.curmodel] is not None:
            self.npoints = self.modelctr.npoints[self.curmodel]
        else:
            self.npoints = None
    
    def switch_model(self,direct):
        if self.modelctr.models[self.curmodel] is not None:
            self.modelctr.models[self.curmodel].to('cpu')
            
        if len(self.modelctr.models) > 1:
            self.curmodel += direct
            if self.curmodel >= len(self.modelctr.models):
                self.curmodel=0
            elif self.curmodel < 0:
                self.curmodel += len(self.modelctr.models)
            self.set_npoints()
                
        if self.modelctr.models[self.curmodel] is not None:
            self.modelctr.models[self.curmodel].to(self.device)
    
    def switch_dataset(self,direct):
        if len(self.datasets) > 1:
            self.curdataset += direct
            if self.curdataset >= len(self.datasets):
                self.curdataset=0
            elif self.curdataset < 0:
                self.curdataset += len(self.datasets)
            self.set_npoints()
            self.datasets_vis_arg_func[self.curdataset](self)
            
        self.curpcl = 0
    
    def switch_pcl(self,direct):
        datalen = self.datasets[self.curdataset].__len__()
        if datalen > 1:
            self.curpcl += direct
            if self.curpcl >= datalen:
                self.curpcl=0
            elif self.curpcl < 0:
                self.curpcl += datalen
        
    
    def get_data(self):
        if self.vis._3d:
            self.vis.changed = True
            pcl,rgb = self.datasets[self.curdataset].__getitem__(self.curpcl)
            
            if self.npoints is None:
                choice = np.arange(pcl.shape[0])
            else:
                choice = np.random.choice(pcl.shape[0],self.npoints,False)
                
            # rgb = pcl[choice,3:6]
            Seg = None
            if self.modelctr.models[self.curmodel] is not None:
                data = self.modelctr.predeal[self.curmodel](pcl[choice,:])
                data = self.modelctr.models[self.curmodel](data)
                Seg,Cls = self.modelctr.resdeal[self.curmodel](data)
            
            self.vis.xyz = pcl[choice,:3]
            if Seg is not None:
                self.vis.mapcolor(pcl.shape[0],self.vis.cmap[Seg,:])
            elif rgb is not None:
                self.vis.mapcolor(pcl.shape[0],pcl[choice,3:6])
            else:
                self.vis.mapcolor(self.vis.xyz.shape[0])
            self.vis.change_pcl_range()
        else:
            data = self.datasets[self.curdataset].__getitem__(self.curpcl)
            if self.modelctr.models[self.curmodel] is not None:
                data = self.modelctr.predeal[self.curmodel](data)
                data = data.to(self.config.device)
                data = self.modelctr.models[self.curmodel](data)
                data =self.modelctr.resdeal[self.curmodel](data)
            self.vis.show = data
        
    def exec(self):
        # getattr(self,self.config.ctrtype)()
        if self.exec_func is not None:
            self.exec_func(self)
   
        pass
    pass