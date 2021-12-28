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
    init = sl.InitParameters()
    init.depth_mode=sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units=sl.UNIT.METER
    init.camera_resolution=sl.RESOLUTION.HD720
    init.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_minimum_distance = 0.15
    init.depth_maximum_distance = 30
    return init
def defaultruntimeparameters():
    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.texture_confidence_threshold = 100
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
    datactr.vis.set(xyzrelist=[1,-1,1],_3d=True)
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
            self.res=res

            self.point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
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
            # vis.changed = True
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            calibration_params = self.zed.get_camera_information().calibration_parameters
            fx = calibration_params.left_cam.fx
            fy = calibration_params.left_cam.fy
            cx = calibration_params.left_cam.cx
            cy = calibration_params.left_cam.cy
            # mtx = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            # disto = calibration_params.left_cam.disto

            cleanpointcloud = self.point_cloud.get_data().astype(np.float32)
            cleanpointcloud.dot(self.tr_np)
            cleanpointcloud = cleanpointcloud.reshape(cleanpointcloud.shape[0]*cleanpointcloud.shape[1],4)
            cleanpointcloud = cleanpointcloud[~np.isnan(cleanpointcloud).any(axis=1)]
            cleanpointcloud = cleanpointcloud[~np.isinf(cleanpointcloud).any(axis=1)]
            # cleanpointcloud[:,3] = ((-1 << 31) - (cleanpointcloud[:,3].astype(np.int64) >> 32 )).astype(np.float32)
            # cleanpointcloud = cleanpointcloud[~np.isnan(cleanpointcloud).any(axis=1)]
            args=np.array([6]).astype(np.float64)
            pointcloud = pcl.endecode_color(cleanpointcloud,args).astype(np.float32)
            # pointcloud = cv2.undistort()

            f = (fx+fy) /2
            x = pointcloud[:,0] - cx
            y = pointcloud[:,1] - cy
            cos_theta = f/np.sqrt(x**2 + y**2 + f ** 2)
            pointcloud[:,2] *= cos_theta
            pointcloud[:,3:6] = np.array([pointcloud[:,5]*10,pointcloud[:,4],pointcloud[:,3]*10]).T
            return pointcloud,True
            # pointcloud[:,1] = 

            # vis.realrgb = np.array([pointcloud[:,5]*10,pointcloud[:,4],pointcloud[:,3]*10]).T
            # if vis.npoints is None:
            #     choice = np.arange(pointcloud.shape[0])
            # else:
            #     choice = np.random.choice(pointcloud.shape[0],vis.npoints,False)
            # if self.model is not None:
            #     seg_pred,_=self.model[self.curmodel].exec(pointcloud[choice,:3+self.model[self.curmodel].normal_channel*3])
            #     seg_pred = torch.argmax(seg_pred.resize(seg_pred.shape[1],seg_pred.shape[2]), dim=1).cpu().data.numpy()
            
            #     vis.xyz = pointcloud[choice,:3]
            #     rgb = vis.cmap[seg_pred,:3]
            #     vis.change_pcl_range()
            #     vis.mapcolor(pointcloud.shape[0],rgb)
            # else:
            #     vis.xyz = pointcloud[:,:3]
            #     vis.mapcolor(pointcloud.shape[0],vis.realrgb[:,:])
            #     vis.change_pcl_range()
                
    # def exec(self):
        
    #     # grab_camera_pcl(self.vis)
    #     self.vis.run_pcl_dsiplay_process()
    #     self.zed.close()

if __name__ == "__main__":
    myzed = zedcamera()
    myzed.exec()



    
