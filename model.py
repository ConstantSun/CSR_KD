import torch.nn as nn
import torch

from student_model import *
from torchvision import models
from utils import save_net,load_net
from collections import OrderedDict

'''
Teacher Net - CSRNet
'''
class CSRNet(nn.Module):
    def __init__(self, batch_norm = False):
    # , load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backend_feat  = [512, 512, 512,256,128,64]
        # self.frontend = make_layers(self.frontend_feat)
        # self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)        
        self.feat = [64, 64, 
                    128, 128, 
                    256, 256, 256,  
                    512, 512, 512,
                    512, 512, 512,256,128,64,
                    64]                            

        
        # vgg = models.vgg16(pretrained = True)

        # self._initialize_weights()


        self.net1 = nn.Sequential( OrderedDict([
            ('conv1',  nn.Conv2d(in_channels = 3, out_channels=self.feat[0], kernel_size = 3, padding = 1, dilation =1)),
            # add or not
            # ('bn1',   nn.BatchNorm2d(self.feat[0])) 
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.resize1 = nn.MaxPool2d(kernel_size=8)
        self.net2 = nn.Sequential(OrderedDict([
            ('conv2',  nn.Conv2d(in_channels = self.feat[0], out_channels=self.feat[1], kernel_size = 3, padding = 1, dilation =1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('mp1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3',  nn.Conv2d(in_channels = self.feat[1], out_channels=self.feat[2], kernel_size = 3, padding = 1, dilation =1)),
            ('relu3', nn.ReLU(inplace=True)),
        ]))
        self.resize2 = nn.MaxPool2d(kernel_size=4)
        self.net3 = nn.Sequential(OrderedDict([
            ('conv4',  nn.Conv2d(in_channels = self.feat[2], out_channels=self.feat[3], kernel_size = 3, padding = 1, dilation =1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('mp2',  nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv5',  nn.Conv2d(in_channels = self.feat[3], out_channels=self.feat[4], kernel_size = 3, padding = 1, dilation =1)),
            ('relu5', nn.ReLU(inplace=True)),            
        ]))
        self.resize3 = nn.MaxPool2d(kernel_size=2)
        self.net4 = nn.Sequential(OrderedDict([
            ('conv6',  nn.Conv2d(in_channels = self.feat[4], out_channels=self.feat[5], kernel_size = 3, padding = 1, dilation =1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7',  nn.Conv2d(in_channels = self.feat[5], out_channels=self.feat[6], kernel_size = 3, padding = 1, dilation =1)),
            ('relu7', nn.ReLU(inplace=True)),        
            ('mp3',  nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv8',  nn.Conv2d(in_channels = self.feat[6], out_channels=self.feat[7], kernel_size = 3, padding = 1, dilation =1)),
            ('relu8', nn.ReLU(inplace=True)),                 
        ]))
        self.net5 = nn.Sequential(OrderedDict([
            ('conv9',  nn.Conv2d(in_channels = self.feat[7], out_channels=self.feat[8], kernel_size = 3, padding = 1, dilation =1)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10',  nn.Conv2d(in_channels = self.feat[8], out_channels=self.feat[9], kernel_size = 3, padding = 1, dilation =1)),
            ('relu10', nn.ReLU(inplace=True)),  
            ('conv11',  nn.Conv2d(in_channels = self.feat[9], out_channels=self.feat[10], kernel_size = 3, padding = 2, dilation =2)),
            ('relu11', nn.ReLU(inplace=True)),                      
        ]))  
        self.net6 = nn.Sequential(OrderedDict([
            ('conv12',  nn.Conv2d(in_channels = self.feat[10], out_channels=self.feat[11], kernel_size = 3, padding = 2, dilation =2)),
            ('relu12', nn.ReLU(inplace=True)),   
            ('conv13',  nn.Conv2d(in_channels = self.feat[11], out_channels=self.feat[12], kernel_size = 3, padding = 2, dilation =2)),
            ('relu13', nn.ReLU(inplace=True)),  
            ('conv14',  nn.Conv2d(in_channels = self.feat[12], out_channels=self.feat[13], kernel_size = 3, padding = 2, dilation =2)),
            ('relu14', nn.ReLU(inplace=True))           
        ]))      

        self.net7 = nn.Sequential(OrderedDict([
            ('conv15',  nn.Conv2d(in_channels = self.feat[13], out_channels=self.feat[14], kernel_size = 3, padding = 2, dilation =2)),
            ('relu15', nn.ReLU(inplace=True)),   
            ('conv16',  nn.Conv2d(in_channels = self.feat[14], out_channels=self.feat[15], kernel_size = 3, padding = 2, dilation =2)),
            ('relu16', nn.ReLU(inplace=True))          
        ]))        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)    
    
        # print("Loading pretrasined weight ...")
        # for k in range(20) : # the first 10 conv layers using pre-trained weight (10 conv + 10 bias -> 20 )
        #     frontend_key = list(self.state_dict())[k]
        #     vgg_key = list(vgg.state_dict())[k]
        #     self.state_dict()[frontend_key].copy_(vgg.state_dict()[vgg_key]) 
            # print("vgg_key: ", vgg_key)
            # print("csr key : ", frontend_key)     

    def forward(self, x: torch.tensor):
        '''
        Later add Batch norm after conv
        '''
        kd_list = []
        resize_list = []
        out = x

        out = self.net1(out)
        kd_list.append(out)
        resize_list.append(self.resize1(out))

        out = self.net2(out)
        kd_list.append(out)
        resize_list.append(self.resize2(out))

        out = self.net3(out)
        kd_list.append(out)
        resize_list.append(self.resize3(out))

        out = self.net4(out)
        kd_list.append(out)
        resize_list.append(kd_list[len(kd_list)-1])

        out = self.net5(out)
        kd_list.append(out)
        resize_list.append(kd_list[len(kd_list)-1])

        out = self.net6(out)
        kd_list.append(out)
        resize_list.append(kd_list[len(kd_list)-1])
        
        out = self.net7(out)           

        x = self.output_layer(out)
        '''
        todo: 
        Using torch.cat here to combine all element in kd_list, not using list. 
        Similarly do that for resize_list.
        '''
        return x, kd_list, resize_list

        
    def _initialize_weights_for_resize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            

def convert_weight1(teacher_model: nn.Module, student_model : nn.Module) -> nn.Module:  
    ''' 
    method1: average according to each axis, then repeat values for that dimmention.
    '''

    for k in range(len(teacher_model.state_dict().items())):
        for l in range(len(student_model.state_dict().items())): 
            teacher_key = list(teacher_model.state_dict())[k]
            student_key = list(student_model.state_dict())[l]
            
            if teacher_key == student_key:
                # print("\n\n\n--------------------------------------------------------------------------")
                # print("key:               ", teacher_key)
                # print("teacher key shape: ", teacher_model.state_dict()[teacher_key].shape)
                # print("student key shape: ", student_model.state_dict()[student_key].shape)
                _s_size = student_model.state_dict()[student_key].shape
                _t_size = teacher_model.state_dict()[teacher_key].shape
                
                weight = 0.
                if len(_s_size) > 1 : 
                    weight = teacher_model.state_dict()[teacher_key].mean(dim=1) #.mean(dim=0)
                    # print("weight shape 1  : ", weight.shape)
                    weight = weight.view(weight.shape[0], 1, weight.shape[1], weight.shape[2])
                    # print("*               : ", weight.shape)
                    weight = weight.repeat(1, _s_size[1], 1, 1)
                    # print("---weight shape 1.1 : ", weight.shape)
                    weight = weight.mean(dim=0)
                    # print("---weight shape 1.2 : ", weight.shape)
                    weight = weight.view(1, weight.shape[0], weight.shape[1], weight.shape[2])

                    weight = weight.repeat(_s_size[0], 1,1,1)
                    # print("---weight shape 1.3 : ", weight.shape)
                else: 
                    weight = teacher_model.state_dict()[teacher_key].mean(dim=0)
                    # print("weight shape 2: ", weight.shape)
                    weight = weight.repeat(_s_size[0])

                # print("weight shape 3: ", weight.shape)
                student_model.state_dict()[student_key].copy_(weight)

    return student_model                


def teacher_loss(): 
    return nn.MSELoss(size_average=False)


# t_model = CSRNet()
# s_model = CSRNet_student(cpr=0.25)


# # print(list(s_model.state_dict())[0])
# s_model = convert_weight1(t_model, s_model)





# from thop import profile
# model = CSRNet()
# input = torch.randn(1,3, 1024, 768)
# macs, params = profile(model, inputs = (input,))   
# print(f"macs: {macs}\nparams: {params}")
# print(f"macs: {macs/10**9} Gi")

# res, kd_list = model(input)

# for i in range(6):
#     print(f"{i}-th element: {kd_list[i].shape}")
# _count = 0
# for i in iter(model.parameters()):
#     _count += 1 
#     print(_count," : ", i.shape)

# # print(_count)
# print(type(model.parameters()))

# print("\nCSRnet: *** *** ***")
# for k, v in model.state_dict().items(): 
#     print(k)
#     print("shape: ", v.shape)
# print("state dict len: \n", len(model.state_dict()))