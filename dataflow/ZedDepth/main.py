import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import cv2
sys.path.append(os.path.abspath("."))
import pyzed.sl as sl
import pclpyd as pcl
from visualizer.OpencvVis import CVis
import importlib
# import datautils

def defaultinit():
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_minimum_distance = 0.15
    init_params.depth_maximum_distance = 30
    return init_params
def defaultruntimeparameters():
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100
    return runtime_parameters
def defaultres():
    res = sl.Resolution()
    res.width = 1920
    res.height = 1080
    return res
def defaulttr_np():
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    return mirror_ref.m

def set_datactr(datactr):
    datactr.vis = CVis(grab_pcl_fuc=datactr.get_data,
                            switch_model_func=datactr.switch_model,
                            switch_dataset_func=datactr.switch_dataset,
                            switch_pcl_func=datactr.switch_pcl,
                            )
    datactr.config.Set_datactr_func=None
    datactr.config.Set_exec_func = datactr.vis.run_pcl_dsiplay_process
    datactr.set()
    
    datactr.curmodel=0
    for i,dataset in enumerate(datactr.config.datatsetstype):
        datactr.set_vis_arg_func.append(importlib.import_module('dataflow.'+dataset+'.main').set_vis_arg)
    datactr.set_vis_arg_func[datactr.curdataset](datactr)
    
def set_vis_arg(datactr):
    datactr.vis.set(_3d=False)
def get_dataset(config):  
    return zedcamera()
    # if modeltypes is not None:
    #     datactr.model = [None]
        
    # datactr.vis = CVis(npoints=50000,xyzrelist=[1,-1,1],ballradius=1,
    #                    grab_pcl_fuc=datactr.data.grab_camera_pcl,
    #                    switch_model_func=datactr.switch_model,
    #                    )

class zedcamera:
    
    def __init__(self,
                 init=defaultinit(),
                 runtime_parameters=defaultruntimeparameters(),
                 res=defaultres(),
                 tr_np=defaulttr_np(),
                 args_set_func=None,):
        if args_set_func is not None:
            args_set_func(self)
        else:
            self.zed = sl.Camera()
            if self.zed is not None:
                self.zed.close()
            # init.coordinate_units = sl.UNIT.METER
            self.init = init
            status = self.zed.open(self.init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                self.zed.close()
                exit()
            self.runtime_parameters = runtime_parameters
            self.res=None

            self.point_cloud = sl.Mat()
            self.image = sl.Mat()
            self.depth = sl.Mat()
            self.tr_np = tr_np
            self.model = [None]
            self.curmodel= 0
            self.npoints = None
    
    def set(self,
            init=None,
            runtime_parameters=None,
            res=None,
            tr_np=None,
            model=[None],
            args_set_func=None,):
        
        # init.coordinate_units = sl.UNIT.METER
        if init is not None:
            self.init = init
            if self.zed is not None:
                self.zed.close()
            
            status = self.zed.open(self.init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
        if runtime_parameters is not None:
            self.runtime_parameters = runtime_parameters
        if res is not None:
            self.res=res
        if tr_np is not None:
            self.tr_np = tr_np
        if model is not None:
            self.model = model
            self.curmodel= len(self.model)-1
        if args_set_func is not None:
            args_set_func(self)
    
    # def switch_model(self,vis):
    #     if len(self.model) == 1:
    #         vis.change_src=False
    #     else:
    #         self.curmodel +=1
    #         if self.curmodel
    def __len__(self):
        return 0
    def __getitem__(self,idx):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

            self.depth_img = self.depth.get_data()
            self.depth_img[np.isnan(self.depth_img)] = 0
            self.depth_img = (self.depth_img * 255/self.depth_img.max()).astype(np.int8)
            return self.depth_img


if __name__ == "__main__":
    myzed = zedcamera()
    myzed.exec()



    
