    def set_plyfile(self,pcd_dir= "../datasets/Linemod_preprocessed/models",xyzrelist=[1,1,1]):
        # pcd_dir = "../datasets/3dvisionpcl/.XYZ"
        files = os.listdir(pcd_dir)
        cloudfiles = []
        for file in files:
            if file[-3:] == 'ply':
                cloudfiles.append(os.path.abspath(os.path.join(pcd_dir,file)))
        self.set_pcl_list(cloudfiles)
        self.grab_pcl_func='grab_ply_pcl'
        self.set_xyz_reverse(xyzrelist)

    def grab_ply_pcl(self):
        if not self.change_src:
            return
        from plyfile import PlyData
        cloudfile = self.curpcllist[self.curpcl]
        cloudmap = PlyData.read(cloudfile)
        cloudmap = convertply2numpy(cloudmap)
        self.xyz = cloudmap[:,0:3]
        rgb = cloudmap[:,3:6]
        self.mapcolor(self.xyz.shape[0],rgb)
        self.change_pcl_range()
        self.change_src = False