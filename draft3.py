import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torchvision, torch 
from torchvision import datasets, transforms
import glob
from PIL import Image
import h5py
from matplotlib import pyplot as plt
import json

# with open('/home/nvhuy/hangdtth/hang/CSR_KD/trancos_training.json', 'r') as outfile:        
#     train_list = json.load(outfile)

# # with open('/home/nvhuy/hangdtth/hang/CSR_KD/shanghai.json', 'r') as outfile:        
# #     train_list = json.load(outfile)

    
# train_loader = dataset.listDataset(train_list,
#                     shuffle=True,
#                     transform=transforms.Compose([
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], # from img net 
#                                                         std=[0.229, 0.224, 0.225]),
#                     ]), 
#                     train=True, 
#                     # seen=model.seen,
#                     batch_size=2,
#                     num_workers=4)                     


# for i, (img, target) in enumerate(train_loader):
    
#     # print(f"img: {img}\ntarget: {target}")
#     print(f"target shape: {target.shape}")
#     print(f"target sum: {target.sum()}")
#     if i == 3 : 
#         break

# import glob 
# l = []
# for f in glob.glob("/home/nvhuy/hangdtth/hang/ShanghaiTech/part_A/train_data/images/*"): 
#     l.append(f)

# # print(l)
# with open('shanghai.json', 'w') as outfile:
#     json.dump(l, outfile)


# img_path = "/home/nvhuy/hangdtth/hang/ShanghaiTech/part_A/train_data/images/IMG_1.jpg"
# gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
# img = Image.open(img_path).convert('RGB')
# gt_file = h5py.File(gt_path, 'r')
# target = np.asarray(gt_file['density'])
# print(f"Target size 1 : {target.shape}")
# target = cv2.resize(target,(int(target.shape[1]/8), int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
# print(f"Target size 2 : {target.shape}")

# target = cv2.resize(target,(int(target.shape[1]/8), int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)
# print(f"Target size 3 : {target.shape}")



# _path = "/home/nvhuy/hangdtth/hang/ShanghaiTech/part_A/test_data"
# ls = []
# for file in glob.glob(_path+"/images/*"): 
#     # img = Image.open(file)
#     # gt_path = file.replace('images', 'ground_truth_h5').replace('jpg', 'h5')
#     # gt_ = h5py.File(gt_path, 'r')
#     # gt = gt_['density']
#     # print(type(gt.value))
#     # # plt.imshow(img, alpha=.5)
#     # # plt.imshow(gt, alpha=.85)
#     # # plt.show()
#     # break
#     ls.append(file)

# with open('shanghai_test.json','w') as outfile:
#     json.dump(ls, outfile)
'''
a
 tensor([[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.]],

         [[ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]],


        [[[17., 18., 19., 20.],
          [21., 22., 23., 24.]],

         [[25., 26., 27., 28.],
          [29., 30., 31., 32.]]]]) 

b
 tensor([[[[33., 34., 35., 36.],
          [37., 38., 39., 40.]],

         [[41., 42., 43., 44.],
          [45., 46., 47., 48.]]],


        [[[49., 50., 51., 52.],
          [53., 54., 55., 56.]],

         [[57., 58., 59., 60.],
          [61., 62., 63., 64.]]]]) 




f1: 
tensor([[[ 1,  2,  3,  4,  5,  6,  7,  8],
         [ 9, 10, 11, 12, 13, 14, 15, 16]],

        [[17, 18, 19, 20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29, 30, 31, 32]]])

f2: tensor([[[33, 34, 35, 36, 37, 38, 39, 40],
         [41, 42, 43, 44, 45, 46, 47, 48]],

        [[49, 50, 51, 52, 53, 54, 55, 56],
         [57, 58, 59, 60, 61, 62, 63, 64]]])

f2: tensor([[[33, 41],
         [34, 42],
         [35, 43],
         [36, 44],
         [37, 45],
         [38, 46],
         [39, 47],
         [40, 48]],

        [[49, 57],
         [50, 58],
         [51, 59],
         [52, 60],
         [53, 61],
         [54, 62],
         [55, 63],
         [56, 64]]])

FSP_matrix:
 tensor([[[ 1356,  1644],
         [ 3692,  4492]],

        [[ 8652,  9964],
         [12012, 13836]]])          
'''          

a = "bc"
b = "cd"
e = "bc"

if a == b :  
  print("a = e")
else :
  print("a =! e")
