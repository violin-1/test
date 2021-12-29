import os
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import cv2
import torch
from PIL import Image
# import model
sys.path.append(".")
import pclpyd as pcl
# import pyzed.sl as sl

sys.path.append("..")
import pclpyd as pcl


def convertply2numpy( src):
    res = np.zeros([src['vertex'].data['x'].shape[0],6])
    for i in range (src['vertex'].data['x'].shape[0]):
        res[i,0] = src['vertex'].data['x'][i]
        res[i,1] = src['vertex'].data['y'][i]
        res[i,2] = src['vertex'].data['z'][i]
        res[i,3] = src['vertex'].data['red'][i]
        res[i,4] = src['vertex'].data['green'][i]
        res[i,5] = src['vertex'].data['blue'][i]
    return res
    
class CVis:
    
    def __init__(self,
                 xyzrelist=[1,-1,1],ballradius=2,grab_pcl_fuc=None,
                 cmap = plt.cm.cool(np.linspace(0,1,50)),background=(0, 0, 0),
                 cvh=128*6,cvw=128*9,cvz=300*3,showrot=False,magnifyBlue=0,
                 freezerot=False,normalizecolor=True,waittime=0,_3d=True,
                 arg_set_func=None,
                 switch_model_func=None,
                 switch_dataset_func=None,
                 switch_pcl_func=None,):

        self.set(xyzrelist,ballradius,grab_pcl_fuc,cmap,background,
                 cvh,cvw,cvz,showrot,magnifyBlue,freezerot,normalizecolor,waittime,_3d,
                 arg_set_func,switch_model_func,switch_dataset_func,switch_pcl_func,)
        self.c_gt=None
        self.c_pred=None
        self.rgb=None
        self.mousex = 0.5
        self.mousey = 0.5
        self.zoom = 1.0
        self.changed = True

        cv2.namedWindow('show3d')
        cv2.moveWindow('show3d', 0, 0)
        cv2.setMouseCallback('show3d', self.onmouse)
        pass
    
    def set(self,
            xyzrelist=None,ballradius=None,grab_pcl_fuc=None,
            cmap = None,background=None,
            cvh=None,cvw=None,cvz=None,showrot=None,magnifyBlue=None,
            freezerot=None,normalizecolor=None,waittime=None,_3d=None,
            arg_set_func=None,
            switch_model_func=None,
            switch_dataset_func=None,
            switch_pcl_func=None,):

        if arg_set_func is not None:
            arg_set_func(self)
        else:
            if cvh is not None :
                self.cvh = cvh
            if cvw is not None :
                self.cvw = cvw
            if cvz is not None :
                self.cvz = cvz
            if freezerot is not None:
                self.freezerot=freezerot
            if normalizecolor is not None:
                self.normalizecolor=normalizecolor
            if waittime is not None:
                self.waittime=waittime
            if showrot is not None:
                self.showrot=showrot
            if magnifyBlue is not None:
                self.magnifyBlue=magnifyBlue
            if ballradius is not None :
                self.ballradius = ballradius
            if background is not None :
                self.background=background
            if grab_pcl_fuc is not None :
                self.grab_pcl_func = grab_pcl_fuc
            if switch_model_func is not None:
                self.switch_model_func=switch_model_func
            if switch_dataset_func is not None:
                self.switch_dataset_func=switch_dataset_func
            if switch_pcl_func is not None:
                self.switch_pcl_func=switch_pcl_func
                
            if xyzrelist is not None :
                self.set_xyz_reverse(xyzrelist)
            if cmap is not None:
                self.cmap = cmap
            if _3d is not None:
                self._3d = _3d
        
    def onmouse(self,*args):
        # global mousex, mousey, changed
        x = args[1]
        y = args[2]
        self.mousex = x / float(self.cvw)
        self.mousey = y / float(self.cvh)
        self.changed = True

    def resample_depth(self, img, w, h, typed='float'):
        imgd = Image.fromarray(img.astype(typed))
        imgd = np.array(imgd.resize((w,h),Image.ANTIALIAS))
        return imgd

    def mapcolor(self,npoints,c_gt=None):
        if self.rgb is None:
            self.rgb = np.zeros((npoints,3), dtype='float32') + 255

        if c_gt is not None:
            # self.c1 = np.zeros((len(self.pointcloud),), dtype='float32') + 255
            # self.c2 = np.zeros((len(self.pointcloud),), dtype='float32') + 255
            self.rgb = np.zeros((c_gt.shape[0],3), dtype='float32') + 255
            self.rgb[:,0] = c_gt[:, 0]
            self.rgb[:,1] = c_gt[:, 1]
            self.rgb[:,2] = c_gt[:, 2]
        else:
            self.rgb = np.zeros((npoints,3), dtype='float32') + 255

        if self.normalizecolor:
            self.rgb[:,0] /= (self.rgb[:,0].max() + 1e-14) / 255.0
            self.rgb[:,1] /= (self.rgb[:,1].max() + 1e-14) / 255.0
            self.rgb[:,2] /= (self.rgb[:,2].max() + 1e-14) / 255.0

        self.rgb[:,0] = np.require(self.rgb[:,0], 'float32', 'C')
        self.rgb[:,1] = np.require(self.rgb[:,1], 'float32', 'C')
        self.rgb[:,2] = np.require(self.rgb[:,2], 'float32', 'C')
    
    def render(self):
        if self.rgb is None:
            return
        rotmat = np.eye(3)
        if not self.freezerot:
            xangle = (self.mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not self.freezerot:
            yangle = (self.mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= self.zoom
        xyz = self.xyz.dot(rotmat) + [self.cvw / 2, self.cvh / 2, -self.cvz/2]

        xyz = np.hstack((xyz,self.rgb)).astype('int32')
        args = np.array([self.cvh,self.cvw,xyz.shape[0],self.ballradius,0]).T

        self.show = np.zeros((self.cvh, self.cvw, 3), dtype='uint8')
        self.show[:] = self.background
        xyz = pcl.render_ball(xyz,args)
        self.show[:,:,0] = xyz[:self.cvh,:]
        self.show[:,:,1] = xyz[self.cvh:2*self.cvh,:]
        self.show[:,:,2] = xyz[2*self.cvh:3*self.cvh,:]
        pass

    def change_pcl_range(self):
        self.xyz = self.xyz - self.xyz.mean(axis=0)
        radius = ((self.xyz ** 2).sum(axis=-1) ** 0.5).max()
        self.xyz[:,0] /= (radius * 2.2) / self.cvw * self.xreverse
        self.xyz[:,1] /= (radius * 1.2) / self.cvh * self.yreverse
        self.xyz[:,2] /= (radius * 2.2) / self.cvz * self.zreverse

    def run_pcl_dsiplay_process(self,datactr=None):
        while True:
            if self.changed or self.grab_pcl_func == 'grab_camera_pcl':
                if self.grab_pcl_func is not None:
                    self.grab_pcl_func()
                if self._3d :
                    self.render()
                self.changed = False
            
            cv2.imshow('show3d', self.show)

            if self.waittime == 0:
                cmd = cv2.waitKey(10) % 256
            else:
                cmd = cv2.waitKey(self.waittime) % 256
            if cmd == ord('q'):
                break
            elif cmd == ord('Q'):
                sys.exit(0)
            if cmd == ord('s') or cmd == ord('S') or cmd == ord('w') or cmd == ord('W'):
                if self.switch_model_func is not None:
                    self.changed = True
                    if cmd == ord('s') or cmd == ord('S'):
                        self.switch_model_func(-1)
                    else:
                        self.switch_model_func(1)
                pass
            if cmd == ord('a') or cmd == ord('A') or cmd == ord('d') or cmd == ord('D'):
                self.changed = True
                if cmd == ord('d') or cmd == ord('D'):
                    self.switch_pcl_func(1)
                else:
                    self.switch_pcl_func(-1)
            if cmd == ord('z') or cmd == ord('Z') or cmd == ord('x') or cmd == ord('X'):
                self.changed = True
                if cmd == ord('x') or cmd == ord('X'):
                    self.switch_dataset_func(1)
                else:
                    self.switch_dataset_func(-1)
                
            if cmd == ord('n'):
                self.zoom *= 1.1
                self.changed = True
            elif cmd == ord('m'):
                self.zoom /= 1.1
                self.changed = True
            elif cmd == ord('r'):
                self.zoom = 1.0
                self.changed = True
            elif cmd == ord('s'):
                cv2.imwrite('show3d.png', self.show)
            if self.waittime != 0:
                break
        return cmd
    pass

    def set_xyz_reverse(self, xyzrelist=[1,1,1]):
        self.xreverse = xyzrelist[0]
        self.yreverse = xyzrelist[1]
        self.zreverse = xyzrelist[2]

        
# if __name__ == "__main__":
#     zed = v()
#     # zed.load_segnet()
#     # zed.load_pointnet2()
#     # zed.load_hut_pointnet2()
#     zed.set_camera(npoints=50000)
#     # zed.set_plyfile()
#     # zed.set_txtfile("/media/violin/32FAD888FAD84A2D/Users/VioLin/Documents/GitHub/kunbo_team/datasets/3D")
#     # zed.set_test_train_file()
#     # zed.set_xyzfile("/media/violin/32FAD888FAD84A2D/Users/VioLin/Documents/GitHub/kunbo_team/datasets/3dvisionpcl/.XYZ")
#     #zed.set_objfile()
#     zed.run_pcl_dsiplay_process()
