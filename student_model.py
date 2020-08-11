import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
from collections import OrderedDict
import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss


class CSRNet_student(nn.Module):
    '''
    Student net - CSRnet student. 
    '''
    def __init__(self, cpr: float, convert_weight = '0'):
        '''
        cpr: channel preservation rate : example: 1 (the same), 1/2 (# of channels = a half of Teacher net), 1/3, 1/4
        '''
        super(CSRNet_student, self).__init__()
        self.seen = 0

        self.feat = [int(64*cpr), int(64*cpr), 
                    int(128*cpr), int(128*cpr), 
                    int(256*cpr), int(256*cpr), int(256*cpr),
                    int(512*cpr), int(512*cpr), int(512*cpr),
                    int(512*cpr),  
                    int(512*cpr), 
                    int(512*cpr), 
                    int(256*cpr),  
                    int(128*cpr), 
                    int(64*cpr),
                    int(64*cpr) ]

        self.net1 = nn.Sequential( OrderedDict([
            ('conv1',  nn.Conv2d(in_channels = 3, out_channels=self.feat[0], kernel_size = 3, padding = 1, dilation =1)),
            # add or not
            # ('bn1',   nn.BatchNorm2d(self.feat[0])) 
            ('relu1', nn.ReLU(inplace=True))
        ]))

        # adjust layer 's purpose:  To make student's dimention equal to teacher's dimention.
        self.adjust1 = nn.Sequential(OrderedDict([
            ('adj1', nn.Conv2d(in_channels = self.feat[0], out_channels=64 , kernel_size=1, padding =0 , dilation =1) )
        ]))
        self.net2 = nn.Sequential(OrderedDict([
            ('conv2',  nn.Conv2d(in_channels = self.feat[0], out_channels=self.feat[1], kernel_size = 3, padding = 1, dilation =1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('mp1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3',  nn.Conv2d(in_channels = self.feat[1], out_channels=self.feat[2], kernel_size = 3, padding = 1, dilation =1)),
            ('relu3', nn.ReLU(inplace=True)),
        ]))
        self.adjust2 = nn.Sequential(OrderedDict([
            ('adj2', nn.Conv2d(in_channels = self.feat[2], out_channels=128 ,kernel_size=1, padding =0 , dilation =1) )
        ]))        
        self.net3 = nn.Sequential(OrderedDict([
            ('conv4',  nn.Conv2d(in_channels = self.feat[2], out_channels=self.feat[3], kernel_size = 3, padding = 1, dilation =1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('mp2',  nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv5',  nn.Conv2d(in_channels = self.feat[3], out_channels=self.feat[4], kernel_size = 3, padding = 1, dilation =1)),
            ('relu5', nn.ReLU(inplace=True)),            
        ]))
        self.adjust3 = nn.Sequential(OrderedDict([
            ('adj3', nn.Conv2d(in_channels = self.feat[4], out_channels=256 ,kernel_size=1, padding =0 , dilation =1) )
        ]))        
        self.net4 = nn.Sequential(OrderedDict([
            ('conv6',  nn.Conv2d(in_channels = self.feat[4], out_channels=self.feat[5], kernel_size = 3, padding = 1, dilation =1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7',  nn.Conv2d(in_channels = self.feat[5], out_channels=self.feat[6], kernel_size = 3, padding = 1, dilation =1)),
            ('relu7', nn.ReLU(inplace=True)),        
            ('mp3',  nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv8',  nn.Conv2d(in_channels = self.feat[6], out_channels=self.feat[7], kernel_size = 3, padding = 1, dilation =1)),
            ('relu8', nn.ReLU(inplace=True)),                 
        ]))
        self.adjust4 = nn.Sequential(OrderedDict([
            ('adj4', nn.Conv2d(in_channels = self.feat[7], out_channels= 512 ,kernel_size=1, padding = 0 , dilation =1) )
        ]))        
        self.net5 = nn.Sequential(OrderedDict([
            ('conv9',  nn.Conv2d(in_channels = self.feat[7], out_channels=self.feat[8], kernel_size = 3, padding = 1, dilation =1)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10',  nn.Conv2d(in_channels = self.feat[8], out_channels=self.feat[9], kernel_size = 3, padding = 1, dilation =1)),
            ('relu10', nn.ReLU(inplace=True)),  
            ('conv11',  nn.Conv2d(in_channels = self.feat[9], out_channels=self.feat[10], kernel_size = 3, padding = 2, dilation =2)),
            ('relu11', nn.ReLU(inplace=True)),                      
        ]))  
        self.adjust5 = nn.Sequential(OrderedDict([
            ('adj5', nn.Conv2d(in_channels = self.feat[10],out_channels=512 , kernel_size=1, padding = 0 , dilation =1) )
        ]))        
        self.net6 = nn.Sequential(OrderedDict([
            ('conv12',  nn.Conv2d(in_channels = self.feat[10], out_channels=self.feat[11], kernel_size = 3, padding = 2, dilation =2)),
            ('relu12', nn.ReLU(inplace=True)),   
            ('conv13',  nn.Conv2d(in_channels = self.feat[11], out_channels=self.feat[12], kernel_size = 3, padding = 2, dilation =2)),
            ('relu13', nn.ReLU(inplace=True)),  
            ('conv14',  nn.Conv2d(in_channels = self.feat[12], out_channels=self.feat[13], kernel_size = 3, padding = 2, dilation =2)),
            ('relu14', nn.ReLU(inplace=True))           
        ]))      
        self.adjust6 = nn.Sequential(OrderedDict([
            ('adj6', nn.Conv2d(in_channels = self.feat[13], out_channels=256 , kernel_size=1, padding = 0 , dilation =1) )
        ]))        

        self.net7 = nn.Sequential(OrderedDict([
            ('conv15',  nn.Conv2d(in_channels = self.feat[13], out_channels=self.feat[14], kernel_size = 3, padding = 2, dilation =2)),
            ('relu15', nn.ReLU(inplace=True)),   
            ('conv16',  nn.Conv2d(in_channels = self.feat[14], out_channels=self.feat[15], kernel_size = 3, padding = 2, dilation =2)),
            ('relu16', nn.ReLU(inplace=True))          
        ])) 
        # Resize layer's purpose is to resize all features in group T (student/teacher list) to the resolution of t4_5 with max pooling.
        self.resize1 = nn.MaxPool2d(kernel_size=8)
        self.resize2 = nn.MaxPool2d(kernel_size=4)
        self.resize3 = nn.MaxPool2d(kernel_size=2)

        self.output_layer = nn.Conv2d(int(64*cpr), 1, kernel_size=1)

        self._initialize_weights_for_adj_layer()

        if convert_weight == '2':
            self.convert_weight2()

    def _initialize_weights_for_adj_layer(self):
        for m in self.state_dict().keys():
            if "adj" in m: 
                if "weight" in m: 
                    nn.init.normal_(self.state_dict()[m], std=0.01)
                else: 
                    nn.init.constant_(self.state_dict()[m], 0)
          
    def convert_weight2(self):                   
        '''
        average pooling over each axis in teacher state dict w.r.t student_model, then transfer into student state dict
        '''
        for m in self.state_dict().keys():
            if "weight" in m: 
                nn.init.normal_(self.state_dict()[m], std=0.01)
            else: 
                nn.init.constant_(self.state_dict()[m], 0)    
    
    def forward(self, x: torch.tensor):
        '''
        Later add Batch norm after conv
        '''
        kd_list = []
        resize_list = []
        out = x

        out = self.net1(out)
        # print("self.net1 shape :", out.shape)
        # print("self.adjust1 shape :", self.adjust1(out).shape)
        kd_list.append(self.adjust1(out))
        resize_list.append(self.resize1(kd_list[len(kd_list)-1]))

        out = self.net2(out)
        kd_list.append(self.adjust2(out))
        resize_list.append(self.resize2(kd_list[len(kd_list)-1]))

        out = self.net3(out)
        kd_list.append(self.adjust3(out))
        resize_list.append(self.resize3(kd_list[len(kd_list)-1]))

        out = self.net4(out)
        kd_list.append(self.adjust4(out))
        resize_list.append(kd_list[len(kd_list)-1])

        out = self.net5(out)
        kd_list.append(self.adjust5(out))
        resize_list.append(kd_list[len(kd_list)-1])

        out = self.net6(out)
        kd_list.append(self.adjust6(out))
        resize_list.append(kd_list[len(kd_list)-1])

        out = self.net7(out)                
        x = self.output_layer(out)

        '''
        todo: 
        Using torch.cat here to combine all element in kd_list, not using list. 
        Similarly do that for resize_list.
        '''
        return x, kd_list, resize_list

            
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

def FSP_matrix(f1: torch.tensor, f2: torch.tensor)  : # (f size: b, c, h, w)
    _b1, _c1, _h1, _w1 = f1.shape[0], f1.shape[1], f1.shape[2], f1.shape[3]
    _b2, _c2, _h2, _w2 = f2.shape[0], f2.shape[1], f2.shape[2], f2.shape[3]

    # print("\nf1 shape: ", f1.shape)
    # print("f2 shape: ", f2.shape)

    return torch.bmm(f1.view(_b1,_c1,_h1*_w1), f2.view(_b2,_c2,_h2*_w2).transpose(1,2))


def triplet_loss(s_kd_list: list, t_kd_list: list, s_resize_list: list, t_resize_list: list, s_out, t_out, target):

    # given 2 tensors size : (100, 64, 128, 96) and (100, 64, 128, 96) (batch size,channel size, width, height)
    """
    s_kd_list: student knowledge distillation list.  - for L_intra
    t_kd_list: teacher knowledge distillation list.  - for L_intra
    s_resize_list: student knowledge distillation after resize - for L_inter
    t_resize_list: teacher knowledge distillation after resize - for L_inter
    s_out: student output_layer.
    t_out: teacher output_layer.
    target: expected output.

    - Calculate L_intra - intra pattern transfer - the similarity of t_i_j and h_i_j at every location (x,y) and every batch  
    Using CosineSimilarity. 

    - Calculate L_inter - inter relationship transfer
    
    - Compute L_m - loss func on density maps (hard ground truth, CSR predicted map, 1/n CSR predicted map)
    
    """

    # print("================================")
    # print("len: ", len(s_kd_list))
    # print(f"length of list:  {len(s_kd_list)}\n\
    #         # of batches: {len(s_kd_list[0])}\n\
    #         channel: {len(s_kd_list[0][0])}\n\
    #         height: {len(s_kd_list[0][0][0])}\n\
    #         width: {len(s_kd_list[0][0][0][0])} ")

    # print(s_kd_list[0].shape)

    L_intra = 0. 
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    _num_elements = len(s_kd_list) # = 6
    
    for i in range(_num_elements): #  Here we have 6 extract layers of shape (b, c, h, w)
        # print("i = ", i+1)
        bchw = s_kd_list[i].shape
        # print("s_kd_list shape : ", bchw)
        # print("t_kd_list shape : ", t_kd_list[i].shape)
        L_intra += 1.*bchw[0]*bchw[2]*bchw[3] - cos(s_kd_list[i],t_kd_list[i]).sum()
                                     
    L_inter = 0. 
    for i in range(_num_elements):  # s_kd_list length = 6
        for j in range(_num_elements):
            if j > i :
                L_inter += torch.norm( FSP_matrix(t_resize_list[i], t_resize_list[j]) - FSP_matrix(s_resize_list[i], s_resize_list[j]) )


    # t_out = t_out.double()
    # s_out = s_out.double()
    L_m = torch.norm(t_out - s_out) + torch.norm(s_out - target)

    print("==================")
    print("L_m: ", L_m)
    print("L_inter: ", L_inter)
    print("L_intra: ", L_intra)    
    print("student output: ", s_out.sum())
    print("Teacher output: ", t_out.sum())

    return L_intra + L_inter + L_m


# from thop import profile
# s_model = CSRNet_student(1./5.)
# t_model = CSRNet()
# input = torch.randn(1,3, 480, 640)  # trancos
# # input = torch.randn(1,3, 1024, 768) # shanghai
# # macs, params = profile(model, inputs = (input,))

# # print(f"macs: {macs}\nparams: {params}")
# # print(f"macs: {macs} ")
# s_res, s_kd_list, s_resize_list = s_model(input)
# t_res, t_kd_list, t_resize_list = t_model(input)

# print(f"{s_res}, s_kd_list, s_resize_list}")
# # print(f"kd list: {len(kd_list)}\nfirst element: {kd_list[0].shape}")
# # print("value\n", res[0][0])
# # for i in range(6):
# #     print(f"{i}-th element: {kd_list[i].shape}")

# optimizer = torch.optim.Adam(model.parameters(), lr=.001)
# loss = triple_loss(kd_list)

# model = CSRNet_student(cpr=.5)
# print("model params: \n")
# for name, param in model.named_parameters():  
#     if param.requires_grad: 
#         print(name)

# t = CSRNet_student(cpr=.5)