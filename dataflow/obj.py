    def set_objfile(self,obj_dir = '../datasets/solidwork',n_sample=100000,xyzrelist=[1,1,1]):
        files = os.listdir(obj_dir)
        cloudfiles = []
        for file in files:
            if file[-3:] == 'STL':
                cloudfiles.append(os.path.abspath(os.path.join(obj_dir,file)))
        self.npoints=n_sample
        self.set_pcl_list(cloudfiles)
        self.grab_pcl_func='grab_obj_pcl'
        self.set_xyz_reverse(xyzrelist)
    
    def grab_obj_pcl(self):
        if not self.change_src:
            return
        args = np.array([self.npoints]).astype(np.float64)
        cloudfile = self.curpcllist[self.curpcl]
        colormap = pcl.readOBJfile([cloudfile],args)
        # for key in cloudmap.keys():
        # self.xyz = colormap["\\".join(colormap.keys().__str__().split('[\'')[1].split('\']')[0].split('\\\\'))][:3]
        pointcloud = colormap[cloudfile].astype(np.float64)
        # face_args = np.array([-500, 500, -500, 500, 0, 1000, 6])
        # pointcloud = pcl.abstract_cube(pointcloud, face_args)
        # pointcloud[:,0] /= 500
        # pointcloud[:,1] /= 500
        # pointcloud[:,2] /= 1000*(-1)
        # if self.model_pred:
        #     npoints = 50000
        #     choice = np.random.choice(pointcloud.shape[0],npoints,False)
        #     seg_pred,_=self.choose_model(pointcloud[choice,:])
        #     seg_pred = torch.argmax(seg_pred.resize(seg_pred.shape[1],seg_pred.shape[2]), dim=1).cpu().data.numpy()
        #     self.xyz = pointcloud[choice,:3]
        #     self.mapcolor(self.xyz.shape[0],self.cmap[seg_pred,:3])
        # else:
        #     self.xyz = pointcloud[:,:3]
        #     self.mapcolor(self.xyz.shape[0])
        self.xyz = pointcloud[:,:3]
        self.mapcolor(self.xyz.shape[0])
        self.change_pcl_range()
        self.change_src = False