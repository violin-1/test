import math

import torch
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
from torch import nn
# from torch.nn import Parameter
# import pdb
# import numpy as np



class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


############################################
#  			Multi-modal 
############################################

class get_model(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):   
        super(get_model, self).__init__()
        
        self.conv1_M1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1_M1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2_M1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3_M1 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.conv1_M2 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1_M2 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2_M2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3_M2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # self.conv1_M3 = nn.Sequential(
        #     basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),    
        # )
        
        # self.Block1_M3 = nn.Sequential(
        #     basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),   
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(196),
        #     nn.ReLU(),  
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),   
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        # )
        
        # self.Block2_M3 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),   
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(196),
        #     nn.ReLU(),  
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),  
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        
        # self.Block3_M3 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),   
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(196),
        #     nn.ReLU(),  
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),   
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        
        self.lastconv1_M1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        self.lastconv1_M2 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        # self.lastconv1_M3 = nn.Sequential(
        #     basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),    
        # )
        
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128*2, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        
        self.lastconv3 = nn.Sequential(
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
        self.fc1 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,2)
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x1, x2):	    	
        
        # RGB
        # x_input = x1
        x = self.conv1_M1(x1)
        
        x = self.Block1_M1(x)	    	    	
        x_Block1 = self.downsample32x32(x)   
        
        x = self.Block2_M1(x)	    
        x_Block2 = self.downsample32x32(x)   
        
        x = self.Block3_M1(x)	    
        x_Block3 = self.downsample32x32(x)  
        
        x_M1 = torch.cat((x_Block1,x_Block2,x_Block3), dim=1) 
        
        # IR
        x = self.conv1_M2(x2)
        
        x = self.Block1_M2(x)	    	    	
        x_Block1 = self.downsample32x32(x)   
        
        x = self.Block2_M2(x)	    
        x_Block2 = self.downsample32x32(x)   
        
        x = self.Block3_M2(x)	    
        x_Block3 = self.downsample32x32(x)  
        
        x_M2 = torch.cat((x_Block1,x_Block2,x_Block3), dim=1)
        
        # Depth
        # x = self.conv1_M3(x3)		   
        
        # x_Block1_M3 = self.Block1_M3(x)	    	    	
        # x_Block1_32x32_M3 = self.downsample32x32(x_Block1_M3)   
        
        # x_Block2_M3 = self.Block2_M3(x_Block1_M3)	    
        # x_Block2_32x32_M3 = self.downsample32x32(x_Block2_M1)   
        
        # x_Block3_M3 = self.Block3_M3(x_Block2_M3)	    
        # x_Block3_32x32_M3 = self.downsample32x32(x_Block3_M3)   
        
        # x_concat_M3 = torch.cat((x_Block1_32x32_M3,x_Block2_32x32_M3,x_Block3_32x32_M3), dim=1)
        

        
        x_M1 = self.lastconv1_M1(x_M1)    
        x_M2 = self.lastconv1_M2(x_M2)    
        # x_M3 = self.lastconv1_M3(x_concat_M3)    
        
        # x = torch.cat((x_M1,x_M2,x_M3), dim=1)
        x = torch.cat((x_M1,x_M2), dim=1)
        
        x = self.lastconv2(x)
        x = self.lastconv3(x)
        
        # map_x = x.squeeze(1)
         
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        # return x,map_x, x_concat_M1, x_Block1_M1, x_Block2_M1, x_Block3_M1, x_input

def get_loss():
    return nn.BCEWithLogitsLoss()
def get_npoints():
    return None
def predeal(data):
    for i in range(data.shape[1]):
        data[:,i] /= data[:,i].max()
    data = torch.Tensor(data[:,:3])
    return data.float().permute(1,0).cuda().view(1,data.shape[1],data.shape[0])
def resdeal(data):
    # data = nn.sigmoid(data)
    ans = torch.argmax(data, dim=1).float()
    return ans

# def contrast_depth_conv(input):
#     ''' compute contrast depth in both of (out, label) '''
#     '''
#         input  32x32
#         output 8x32x32
#     '''
    

#     kernel_filter_list =[
#                         [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
#                         [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
#                         [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
#                         ]
    
#     kernel_filter = np.array(kernel_filter_list, np.float32)
    
#     kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
#     # weights (in_channel, out_channel, kernel, kernel)
#     kernel_filter = kernel_filter.unsqueeze(dim=1)
    
#     input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
#     contrast_depth = torch.nn.functional.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
#     return contrast_depth

# class Contrast_depth_loss(torch.nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
#     def __init__(self):
#         super(Contrast_depth_loss,self).__init__()
#         return
#     def forward(self, out, label): 
#         '''
#         compute contrast depth in both of (out, label),
#         then get the loss of them
#         tf.atrous_convd match tf-versions: 1.4
#         '''
#         contrast_out = contrast_depth_conv(out)
#         contrast_label = contrast_depth_conv(label)
        
        
#         criterion_MSE = torch.nn.MSELoss().cuda()
    
#         loss = criterion_MSE(contrast_out, contrast_label)
#         #loss = torch.pow(contrast_out - contrast_label, 2)
#         #loss = torch.mean(loss)
    
#         return loss