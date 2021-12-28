import os
import numpy as np
import sys
import torch
# import model
sys.path.append(os.path.abspath("."))
import pclpyd as pcl
from visualizer.OpencvVis import CVis
import importlib
# pcd_dir = "../datasets/3dvisionpcl/.XYZ"
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
    datactr.vis.set(xyzrelist=[1,1,-1],_3d=True)
def get_dataset(config):
    return XYZ3dvision()

class XYZ3dvision:
    
    def __init__(self):
        self.datadir = "/media/violin/32FAD888FAD84A2D/Users/VioLin/Documents/GitHub/kunbo_team/datasets/3dvisionpcl/.XYZ"
        self.files = os.listdir(self.datadir)
        self.cloudfiles = []
        for file in self.files:
            if file[-3:] == 'xyz':
                self.cloudfiles.append(os.path.abspath(
                    os.path.join(self.datadir, file)))
        self.cloudfiles.sort()
        self.npoints = 20000
        pass

    def __len__(self):
        return len(self.cloudfiles)
        pass

    def __getitem__(self, idx):
        args = np.array([6, 6]).astype(np.float64)
        cloudfile = self.cloudfiles[idx]
        colormap = pcl.readPCLfile([cloudfile], args)
        pointcloud = colormap[cloudfile].astype(np.float64)
        face_args = np.array([-500, 500, -500, 500, 0, 1000, 6])
        pointcloud = pcl.abstract_cube(pointcloud, face_args)
        return pointcloud,None
        pass

# datadir="/media/violin/32FAD888FAD84A2D/Users/VioLin/Documents/GitHub/kunbo_team/datasets/3dvisionpcl/.XYZ"

# def get_data():
#     files = os.listdir(datadir)
#     cloudfiles = []
#     for file in files:
#         if file[-3:] == 'xyz':
#             cloudfiles.append(os.path.abspath(os.path.join(datadir,file)))
#     cloudfiles.sort()
#     vis.set(lencurpcl=len(cloudfiles),npoints=20000,xyzrelist=[1,1,-1],ballradius=1,grab_pcl_fuc=grab_xyz_pcl)

# def grab_xyz_pcl(vis):
#     if not vis.change_src:
#         return
#     args = np.array([6,6]).astype(np.float64)
#     cloudfile = vis.curpcllist[vis.curpcl]
#     colormap = pcl.readPCLfile([cloudfile],args)
#     # for key in cloudmap.keys():
#     # vis.xyz = colormap["\\".join(colormap.keys().__str__().split('[\'')[1].split('\']')[0].split('\\\\'))][:3]
#     pointcloud = colormap[cloudfile].astype(np.float64)
#     face_args = np.array([-500, 500, -500, 500, 0, 1000, 6])
#     pointcloud = pcl.abstract_cube(pointcloud, face_args)
#     # pointcloud[:,0] /= 500
#     # pointcloud[:,1] /= 500
#     # pointcloud[:,2] /= 1000*(-1)
#     choice = np.random.choice(pointcloud.shape[0],vis.npoints,False)
#     if vis.model_pred:
#         xyz = pointcloud[choice,:vis.channel]
#         for i in range(vis.channel):
#             xyz[:,i] /= xyz[:,i].max()
#         seg_pred,_=vis.choose_model(xyz)
#         seg_pred = torch.argmax(seg_pred.resize(seg_pred.shape[1],seg_pred.shape[2]), dim=1).cpu().data.numpy()
#         vis.xyz = pointcloud[choice,:3]
#         vis.mapcolor(vis.xyz.shape[0],vis.cmap[seg_pred,:3])
#     else:
#         vis.xyz = pointcloud[choice,:3]
#         vis.mapcolor(vis.xyz.shape[0])
#     vis.change_pcl_range()
#     vis.change_src = False
    

# vis = CVis()
# set_xyzfile(vis)
# # grab_camera_pcl(vis)
# vis.run_pcl_dsiplay_process() 
    