import torch
import torch.nn as nn
import numpy as np


# a = torch.range(1, 32, 1).view(2,2,2,4)
# print(a,'\n', a.dtype)


# b= torch.range(33,64,1).view(2,2,2,4)
# print(b,'\n', b.dtype)

# # cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# # print("=======================")


# # print(cos(a,b))
# # print(cos(a,b).sum())
# # a[range(2)]
# # a = torch.range(1,16).view(2,2,4)
# # b= torch.range(17,32).view(2,2,4)
# a.requires_grad_(True)
# b.requires_grad_(True)

# c= torch.range(1,6).view(2,3)
# c.requires_grad_(True)
# d = torch.range(7,12).view(2,3)
# d.requires_grad_(True)
# # print("c : \n",c)
# # print("d : \n",d)
# # print(torch.dot(a[range(2),range(2)], b[range(2),range(2)]))
# print("a : \n",a)
# print("b : \n",b)
# print("testing")
# tem = a*b
# print(tem)
# print(tem.requires_grad)



# a = torch.arange(1,17).view(2,2,4)
# b = torch.arange(17,41).view(3,2,4)

# print("true value: ")
# print(torch.sum(a[0]*b[0]))
# print(torch.sum(a[0]*b[1]))
# print(torch.sum(a[0]*b[2]))

# print("a : \n",a)
# print("b : \n",b)

# a = a.reshape(2,8)
# b = torch.transpose(b.reshape(3,8),0,1)

# print("test1: ")
# print("reshaping :")
# print(a)
# print(b)
# print("============")

# print(a@b)

# print("===============")

# import numpy as np
# a = np.array([[0, 1, 2],
#               [4, 5, 6]])

# b = np.array(
# [
#     [0, 1, 2],
#     [4, 5, 6],
#     [-1, 0, 1],
#     [-3, -2, 1]
# ])

# print(np.einsum('ij,hj->hi', a, b))

# a = torch.arange(1, 33, 1).view(2,2,2,4)
# print( 'a\n', a,'\n', a.dtype)

# b= torch.arange(33,65,1).view(2,2,2,4)
# print( 'b\n',  b,'\n', b.dtype)

# def FSP_matrix(f1: torch.tensor, f2: torch.tensor)  : # (f size: b, c, h, w)
#     _b, _c, _h, _w = f1.shape[0], f1.shape[1], f1.shape[2], f1.shape[3]
#     # f1 = f1.view(_b,_c,_h*_w)
#     # f2 = f2.view(_b,_c,_h*_w)
#     # print(f"f1: \n{f1}\n\nf2: {f2}\n")

#     # f2 = f2.transpose(1,2)
#     # print(f"f2: {f2}\n")

#     # return torch.bmm(f1,f2)
#     return torch.bmm(f1.view(_b,_c,_h*_w), f2.view(_b,_c,_h*_w).transpose(1,2))

# print("FSP_matrix:\n", FSP_matrix(a, b))

# b = b.view(2,16)
# b = b.transpose(0, 1)
# print("b tranpose: ", b)


# a = torch.arange(0,36).view(2,2,3,3).type('torch.DoubleTensor')
# print("a:\n", a)
# print(a.mean(dim=1))
# print(a.mean(dim=1).mean(dim=0))

x = torch.arange(0,24).view(4,2,3)
x = x.type(torch.DoubleTensor)
print("1:\n", x)
print("2:\n", x.dtype)
# print(x.repeat(2,2,2,3))
# print(x.mean(dim=0))
print(x.repeat(2,1,1))


