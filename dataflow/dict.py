 
    def set_dict_pcl(self, cloudmap, npoints=20000,cmap=None,xyzrelist=[1,1,1]):
        cloudfiles = []
        for key in cloudmap.keys():
            cloudfiles.append(key)
        self.pcl_map = cloudmap
        self.set_pcl_list(cloudfiles)        
        self.grab_pcl_func='grab_dict_pcl'
        self.npoints = npoints
        self.set_xyz_reverse(xyzrelist)
        if cmap is None:
            
            self.cmap = np.array(plt.cm.cmap_d['viridis'].colors)#plt.cm.cool(np.linspace(0,1,2))
        else:
            self.cmap = cmap
    
    def grab_dict_pcl(self):
        cloudfile = self.curpcllist[self.curpcl]
        cloud = self.pcl_map[cloudfile]
        choice=None
        if cloud.shape[0] > self.npoints:
            choice = np.random.choice(cloud.shape[0],self.npoints,False)
        if choice is None:
            self.xyz = cloud[:,:3]
        else:
            self.xyz = cloud[choice,:3]

        if cloud.shape[1] ==3:
            self.mapcolor(self.xyz.shape[0])
        elif cloud.shape[1] == 6:
            self.mapcolor(self.xyz.shape[0],cloud[choice,3:6])
        elif cloud.shape[1] == 4:
            self.mapcolor(self.xyz.shape[0],self.cmap[cloud[choice,3].astype(np.uint8),:3])
        
        self.change_pcl_range()
        self.change_src = False