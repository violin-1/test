import os
import numpy as np
import random

def identify(root):
    if 'live' in root or '真人' in root:
        return True
    return False

def findsamename(root):
    dolpfiles = os.listdir(os.path.join(root,'DOLP'))
    s0files = os.listdir(os.path.join(root,'S0'))
    s0flag = False
    if '.png_s0' in s0files[0]:
        s0flag=True
        _s0files = []
        for s0file in s0files:
            temp = s0file.split('.png_s0')
            _s0files.append(temp[0]+temp[1])
        s0files = _s0files
    
    dolpflag = False
    if '.png_dolp' in dolpfiles[0]:
        dolpflag = True
        _dolpfiles = []
        for dolpfile in dolpfiles:
            temp = dolpfile.split('.png_dolp')
            _dolpfiles.append(temp[0]+temp[1])
        dolpfiles = _dolpfiles
          
    dolpset,s0set = set(dolpfiles),set(s0files)
    subfiles = list(dolpset&s0set)
    
    
    _s0files = []
    for s0file in subfiles:
        if s0flag:
            temp = s0file.split('.')
            name = os.path.join(root,'S0',temp[0]+'.png_s0.'+temp[1])
            _s0files.append((name,int(identify(name))))
        else:
            name = os.path.join(root,'S0',s0file)
            _s0files.append((name,int(identify(name))))
    s0files = np.array(_s0files)
    
    _dolpfiles = []
    for dolpfile in subfiles:
        if dolpflag:
            temp = dolpfile.split('.')
            name = os.path.join(root,'DOLP',temp[0]+'.png_dolp.'+temp[1])
            _dolpfiles.append((name,int(identify(name))))
        else:
            name = os.path.join(root,'DOLP',dolpfile)
            _dolpfiles.append((name,int(identify(name))))
    dolpfiles = np.array(_dolpfiles)
    dolpreal,dolpfake,s0real,s0fake = [],[],[],[]
    ind_ = np.where(dolpfiles[:,1]=='0')[0]
    ind = np.where(dolpfiles[:,1]=='1')[0]
    dolpreal = list(dolpfiles[ind])
    # dolpfake = list(dolpfiles[ind_])
    s0real = list(s0files[ind])
    # s0fake = list(s0files[ind_])
    
    return dolpreal,s0real

def searchalldata(root,flag=False):
    dolpreal,s0real = [],[]
    if flag:
        items = []
        items.append(os.path.join('dataset1_live','HUT','Only_Face'))
        # items.append(os.path.join('dataset2_all'))
        # items.append(os.path.join('dataset3_attack','HUT','Only_Face'))
    else:
        items = os.listdir(root)
    for item in items:
        # if item.__len__() <=1 or item == 'Deg' or '.txt' in item or 'S2' in item or 'S3' in item:
        #     continue
        if '.' in item:
            if identify(root):
                if 'DOLP' in root:
                    dolpreal.append((os.path.join(root,item),1))
                elif 'S0' in root:
                    s0real.append((os.path.join(root,item),1))
            # else:
            #     if 'DOLP' in root:
            #         dolpfake.append((os.path.join(root,item),0))
            #     elif 'S0' in root:
            #         s0fake.append((os.path.join(root,item),0))
        else:
            if 'dataset3_attack' in item or 'HUT' in item:
                tmp1,tmp2 = findsamename(os.path.join(root,item))
                dolpreal = dolpreal + tmp1
                # dolpfake = dolpfake + tmp2
                s0real = s0real + tmp2
                # s0fake = s0fake + tmp4
            else:
                # tmp1,tmp2,tmp3,tmp4 = searchalldata(os.path.join(root,item))
                tmp1,tmp2 = searchalldata(os.path.join(root,item))
                dolpreal = dolpreal + tmp1
                # dolpfake = dolpfake + tmp2
                s0real = s0real + tmp2
                # s0fake = s0fake + tmp4
    return dolpreal,s0real

def shuffle_split_data(real,testratio=0.2):
    random.shuffle(real)
    # random.shuffle(fake)
    len1 = len(real)
    # len2 = len(fake)
    tlen1 = int(len1*testratio)
    # tlen2 = int(len2*testratio)
    
    realtest = real[:tlen1]
    realtrain = real[tlen1:]

    return realtrain,realtest

def gendatalist(root,testratio=0.2):
    dolpreal,s0real = searchalldata(root,True)
    indreal = np.arange(0,len(dolpreal))
    # indfake = np.arange(0,len(dolpfake))
    indrealtrain,indrealtest = shuffle_split_data(indreal,testratio)
    dolpreal,s0real = np.array(dolpreal),np.array(s0real)
    dolptraindir,dolptestdir = dolpreal[indrealtrain],dolpreal[indrealtest]
    s0traindir,s0testdir = s0real[indrealtrain],s0real[indrealtest]
    return dolptraindir,dolptestdir,s0traindir,s0testdir

def save_train_test_to_txt(traindir,testdir,txtsavepath,filenameprefix):
    # traindir,testdir=gendatalist(datadir)
    # with open(os.path.join(txtsavepath,filenameprefix+'all.txt'),'w') as f:
    #     all_ = traindir + testdir
    #     for line in all_:
    #         f.write(line[0]+','+str(line[1])+'\n')
            
    with open(os.path.join(txtsavepath,filenameprefix+'vis.txt'),'w') as f:
        for line in traindir:
            f.write(line[0]+','+str(line[1])+'\n')
    
    with open(os.path.join(txtsavepath,filenameprefix+'vis.txt'),'a+') as f:
        for line in testdir:
            f.write(line[0]+','+str(line[1])+'\n')

def change(root):
    root = os.path.join(root,'dataset2_all')
    types = os.listdir(root)
    for typ in types:
        if typ == '1' or typ == '2':
            continue;
        else:
            x1 =  os.listdir(os.path.join(root,typ,'DOLP'))
            
            x2 =  os.listdir(os.path.join(root,typ,'S0'))
            if '真人' in typ:
                x1.sort(key= lambda x:int(x[x.find('(')+1:x.find(')')]))
                x2.sort(key= lambda x:int(x[x.find('(')+1:x.find(')')]))
            for i in range(len(x1)-1,-1,-1):
                if x1[i] == x2[i]:
                    continue
                os.rename(os.path.join(root,typ,'S0',x2[i]),os.path.join(root,typ,'S0',x1[i]))
            pass
    pass
     
if __name__ == '__main__':
    datadir = os.path.join(os.path.abspath('.'),'datasets','multi_modal_Polarization')
    mode = 'gen'
    if mode == 'gen':
        dolptraindir,dolptestdir,s0traindir,s0testdir = gendatalist(datadir,0.3)
        save_train_test_to_txt(dolptraindir,dolptestdir,datadir,'gen_DOLP_')
        save_train_test_to_txt(s0traindir,s0testdir,datadir,'gen_S0_')     
    elif mode == 'change':
        change(datadir)
    pass