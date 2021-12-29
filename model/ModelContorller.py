# import matplotlib.pyplot as plt
import numpy as np
import torch
import importlib
import os

# modelname='segnet',modelfiledir='model/vanilla_segmentation/trained_models', modelfilename='model_58_0.12319542670436204.pth',mode='gpu'
class ModelController:
    
    def __init__(self,
                modeltypes=['pointnet2_part_seg_msg'],
                arg_set_func=None):
        self.set(modeltypes,arg_set_func)
        # if arg_set_func is not None:
        #     arg_set_func(self)
        # else:
        #     # self.cmap = plt.cm.hot(np.linspace(0,1,50))
        #     self.models = [None]
        #     self.npoints = [None]
        #     self.predeal = [None]
        #     self.resdeal = [None]
        #     if modeltypes is not None:
        #         for i, modeltype in enumerate(modeltypes):
        #             # self.modelname = modelname
        #             self.models.append(importlib.import_module('model.'+modeltype+'.main').get_model())
        #             self.npoints.append(importlib.import_module('model.'+modeltype+'.main').get_npoints())
        #             self.predeal.append(importlib.import_module('model.'+modeltype+'.main').predeal)
        #             self.resdeal.append(importlib.import_module('model.'+modeltype+'.main').resdeal)
        #         # self.mode = mode
        #         # if self.mode == 'gpu' or self.mode == 'cuda':
        #         #     self.model = self.model.cuda()
        #             checkpoint = torch.load('model/{0}/checkpoints/{1}'.format(modeltype,modelfilename))
        #             self.models[i+1].load_state_dict(checkpoint['model_state_dict'])
        #     # self.data_predel_func = data_predel_func
        pass

    def set(self,
            modeltypes=None,
            arg_set_func=None):
        if arg_set_func is not None:
            arg_set_func(self)
        else:
            # self.cmap = plt.cm.hot(np.linspace(0,1,50))
            if modeltypes is not None:
                self.models = [None]
                self.npoints = [None]
                self.predeal = [None]
                self.resdeal = [None]
                self.lossfuc = [None]
                self.checkpoints = [None]
                if modeltypes is not None:
                    for i, modeltype in enumerate(modeltypes):
                        # self.modelname = modelname
                        self.models.append(importlib.import_module('model.'+modeltype+'.main').get_model())
                        self.npoints.append(importlib.import_module('model.'+modeltype+'.main').get_npoints())
                        self.predeal.append(importlib.import_module('model.'+modeltype+'.main').predeal)
                        self.resdeal.append(importlib.import_module('model.'+modeltype+'.main').resdeal)
                        self.lossfuc.append(importlib.import_module('model.'+modeltype+'.main').get_loss())
                    # self.mode = mode
                    # if self.mode == 'gpu' or self.mode == 'cuda':
                    #     self.model = self.model.cuda()
                        if os.path.exists('datasets/datasetp/{0}/checkpoints/best.pth'.format(modeltype)):
                            checkpoint = torch.load('datasets/datasetp/{0}/checkpoints/best.pth'.format(modeltype))
                            self.checkpoints.append(checkpoint)
                            self.models[i+1].load_state_dict(checkpoint['state_dict'])
                        else:
                            self.checkpoints.append(None)
            # self.data_predel_func = data_predel_func
        pass
        
    # def exec(self,data):
    #     torch.cuda.empty_cache()
    #     with torch.no_grad():
    #         res = self.model(data)
    #     return res
        pass