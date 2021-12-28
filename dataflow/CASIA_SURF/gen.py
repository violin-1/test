import os
import numpy as np
import random

def identify(root):
    if 'real' in root:
        return True
    return False

def searchalldata(root,flag=False):
    datadict = {}
    datadict['depthreal'] = []
    datadict['depthfake'] = []
    datadict['rgbreal'] = []
    datadict['rgbfake'] = []
    datadict['irreal'] = []
    datadict['irfake'] = []
    datadict['label'] = []

    if flag:
        items = []
        items.append(os.path.join(root,'Training'))
    else:
        items = os.listdir(root)
    for item in items:
        # if item.__len__() <=1 or item == 'Deg' or '.txt' in item or 'S2' in item or 'S3' in item:
        #     continue
        if '.jpg' in item:
            if identify(root):
                if 'depth' in root:
                    datadict['depthreal'].append(os.path.join(root,item))
                elif 'color' in root:
                    datadict['rgbreal'].append(os.path.join(root,item))
                elif 'ir' in root:
                    datadict['irreal'].append(os.path.join(root,item))
            else:
                if 'depth' in root:
                    datadict['depthfake'].append(os.path.join(root,item))
                elif 'color' in root:
                    datadict['rgbfake'].append(os.path.join(root,item))
                elif 'ir' in root:
                    datadict['irfake'].append(os.path.join(root,item))
        else:
            datadictsub = searchalldata(os.path.join(root,item))
            datadict['depthreal'] += datadictsub['depthreal']
            datadict['depthfake'] += datadictsub['depthfake']
            datadict['rgbreal'] += datadictsub['rgbreal']
            datadict['rgbfake'] += datadictsub['rgbfake']
            datadict['irreal'] += datadictsub['irreal']
            datadict['irfake'] += datadictsub['irfake']
    return datadict

def shuffle_split_data(real,fake,testratio=0.2):
    random.shuffle(real)
    random.shuffle(fake)
    len1 = len(real)
    len2 = len(fake)
    tlen1 = int(len1*testratio)
    tlen2 = int(len2*testratio)
    
    realtest,faketest = real[:tlen1],fake[:tlen2]
    realtrain,faketrain = real[tlen1:],fake[tlen2:]

    return realtrain,faketrain,realtest,faketest

def gendatalist(root,trainnum=1500):
    datadict = searchalldata(root,True)
    
    data = {}
    k = 0
    for i,(depth,color,ir) in enumerate(zip(datadict['depthreal'],datadict['rgbreal'],datadict['irreal'])):
        tmp = {}
        tmp['depth'] = depth
        tmp['rgb'] = color
        tmp['ir'] = ir
        tmp['label'] = 1
        data[k+i] = tmp
        
    k=len(data.keys())
    for i,(depth,color,ir) in enumerate(zip(datadict['depthfake'],datadict['rgbfake'],datadict['irfake'])):
        tmp = {}
        tmp['depth'] = depth
        tmp['rgb'] = color
        tmp['ir'] = ir
        tmp['label'] = 0
        data[k+i] = tmp
    
    # path = os.path.join(root,'data','val_private_list.txt')
    
    import fileinput
    k = len(data.keys())
    for i,line in enumerate(fileinput.input(os.path.join(root,"val_private_list.txt"))):
        xlist = line.split(' ')
        tmp = {}
        tmp['rgb'] = os.path.join(root,xlist[0])
        tmp['depth'] = os.path.join(root,xlist[1])
        tmp['ir'] = os.path.join(root,xlist[2])
        tmp['label'] = xlist[3][0]
        data[k+i] = tmp
        
    ind = np.array(range(len(data.keys())))
    random.shuffle(ind)
    
    trainInd = ind[:trainnum]
    testInd = ind[trainnum:]
    
    traindata = np.array(list(data.values()))[trainInd]
    testdata = np.array(list(data.values()))[testInd]
    
    return traindata,testdata

def save_train_test_to_txt(traindir,testdir,txtsavepath,filenameprefix):
    # traindir,testdir=gendatalist(datadir)
    # with open(os.path.join(txtsavepath,filenameprefix+'all.txt'),'w') as f:
    #     all_ = traindir + testdir
    #     for line in all_:
    #         f.write(line[0]+','+str(line[1])+'\n')
            
    with open(os.path.join(txtsavepath,filenameprefix+'train.txt'),'w') as f:
        for line in traindir:
            f.write(line['rgb']+','+line['depth']+','+line['ir']+','+str(line['label'])+'\n')
    
    with open(os.path.join(txtsavepath,filenameprefix+'test.txt'),'w') as f:
        for line in testdir:
            f.write(line['rgb']+','+line['depth']+','+line['ir']+','+str(line['label'])+'\n')

def collect_all_data(root,testnum=1500):
    
    pass

if __name__ == '__main__':
    datadir = os.path.join(os.path.abspath('.'),'datasets','CASIA_SURF')
    mode = 'gen'
    if mode == 'gen':
        traindata,testdata = gendatalist(datadir)
        save_train_test_to_txt(traindata,testdata,datadir,'Mytest_')  

    pass