def set_txtfile(self,txt_dir = '../datasets/3dvisionpcl/3D',npoints = 20000,xyzrelist=[1,1,-1]):
        sub_dirs = os.listdir(txt_dir)
        
        cloudfiles = []
        for sub_dir in sub_dirs:
            files = os.listdir(os.path.abspath(os.path.join(txt_dir,sub_dir)))
            for file in files:
                if file[-3:] == 'txt':
                    cloudfiles.append(os.path.abspath(os.path.join(txt_dir,sub_dir,file)))
        self.npoints = npoints
        self.set_pcl_list(cloudfiles)
        self.pcltype=6
        self.grab_pcl_func='grab_txt_pcl'
        self.set_xyz_reverse(xyzrelist)
    
    def set_test_train_file(self,txt_dir='/data1/yalin.huang/3D_/',filename = 'test.txt',npoints=5000,xyzrelist=[-1,1,-1]):
        cloudfiles = []
        f = open(os.path.join(txt_dir,filename))
        line = f.readlixne()
        while line:
            cloudfiles.append(line[:-1])
            line = f.readline()
        self.npoints = npoints
        self.set_pcl_list(cloudfiles)
        self.pcltype=6
        self.grab_pcl_func='grab_txt_pcl'
        self.set_xyz_reverse(xyzrelist)

    def grab_txt_pcl(self):
        if not self.change_src:
            return
        args = np.array([6,self.pcltype]).astype(np.float64)
        cloudfile = self.curpcllist[self.curpcl]
        print(cloudfile)
        colormap = pcl.readPCLfile([cloudfile],args)
        pointcloud = colormap[cloudfile].astype(np.float64)

        if self.model_pred:
            choice = np.random.choice(pointcloud.shape[0],self.npoints,False)
            xyz = pointcloud[choice,:self.channel]
            for i in range(self.channel):
                xyz[:,i] /= xyz[:,i].max()
            seg_pred,_=self.choose_model(xyz)
            seg_pred = torch.argmax(seg_pred.resize(seg_pred.shape[1],seg_pred.shape[2]), dim=1).cpu().data.numpy()
            self.xyz = pointcloud[choice,:3]
            self.mapcolor(self.xyz.shape[0],self.cmap[seg_pred,:3])
        else:
            # face_args = np.array([-1000, 1000, -1000, 1000, 0, 2200,6])
            # face_point = pcl.abstract_cube(pointcloud, face_args)
            face_point = pointcloud
            choice = np.random.choice(face_point.shape[0],self.npoints,False)
            self.xyz = face_point[choice,:3]
            self.xyz[:,0] /= self.xyz[:,0].max()
            self.xyz[:,1] /= self.xyz[:,1].max()
            self.xyz[:,2] /= self.xyz[:,2].max()

            self.mapcolor(self.xyz.shape[0],face_point[choice,3:6])       
        # self.mapcolor(self.xyz.shape[0])
        self.change_pcl_range()
        self.change_src = False
    