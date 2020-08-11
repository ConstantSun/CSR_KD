import sys
import os
from student_model import *

import warnings

from model import *

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import argparse
import json
import cv2
import dataset
import time

"""
TRANCOS V3

python train.py /home/nvhuy/hangdtth/hang/CSR_KD/trancos_training.json  /home/nvhuy/hangdtth/hang/CSR_KD/trancos_test.json 0 0
training set:  404 images
validation  :  421 images
train val set: 824 images
test      set: 422 images
"""

"""
SHANGHAI CROWNED

python train.py /home/nvhuy/hangdtth/hang/CSR_KD/shanghai_train.json  /home/nvhuy/hangdtth/hang/CSR_KD/shanghai_test.json --pre /home/nvhuy/hangdtth/hang/CSR_KD/PartAmodel_best.pth 0 0

train set : 300
test set  : 182
"""


parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', default=0, type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', default=0, type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.cpr = 0.25
    args.original_lr = 1e-9
    args.lr = 1e-9
    args.batch_size    = 1

    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epoches        = 20
    args.steps         = [-1,1,50,100]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 1

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    teacher_model = CSRNet()
    student_model = CSRNet_student(cpr = args.cpr)
    
    teacher_model = teacher_model.cuda()
    student_model = student_model.cuda()
    _student_origin = CSRNet_student(cpr = args.cpr).cuda()

    for k in range(len(student_model.state_dict().items())):
        student_key = list(student_model.state_dict())[k]
        original_ = list(_student_origin.state_dict())[k]
        _student_origin.state_dict()[original_].copy_(student_model.state_dict()[student_key])

    
    criterion = triplet_loss
    
    # optimizer = torch.optim.SGD( 
    #                              # itertools.chain(),
    #                             student_model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.decay)
    optimizer = torch.optim.Adam(student_model.parameters(), 
                    lr=args.original_lr, betas=(0.9, 0.999), eps=1e-1, weight_decay=0, amsgrad=False)


    '''
    Loading pretrained weight for teacher model
    '''
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            for k in range(len(teacher_model.state_dict().items())):
                teacher_key = list(teacher_model.state_dict())[k]
                checkpoint_key = list(checkpoint['state_dict'])[k]
                teacher_model.state_dict()[teacher_key].copy_(checkpoint['state_dict'][checkpoint_key]) 

            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch']) )            
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    '''
    Convert weight from teacher to student (3 methods)
    '''
    student_model = convert_weight1(teacher_model, student_model)

    # print("testing ...")
    # for k in range(len(student_model.state_dict().items())):
    #     student_key = list(student_model.state_dict())[k]
    #     original_ = list(_student_origin.state_dict())[k]
    #     print((student_model.state_dict()[student_key] - _student_origin.state_dict()[original_]).sum() )


    for epoch in range(args.epoches):
        
        # adjust_learning_rate(optimizer, epoch)
        
        train(train_list, teacher_model, student_model, criterion, optimizer, epoch)

        prec1 = validate(val_list, student_model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': student_model.state_dict(),
            'best_prec1': best_prec1,
            # 'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)



def train(train_list, teacher_model, student_model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    _train_loader = dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], # from img net ?
                                                                             std=[0.229, 0.224, 0.225]),
                                                        ]), 
                            train=True, 
                            # seen=model.seen,
                            )   
                      
    train_loader = DataLoader( _train_loader, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)   
    
    print(f'epoch {epoch}, processed {(train_loader.__len__)} samples, lr {args.lr} \n\n' )

    student_model.train()
    end = time.time()
    # print(f"train loader dataset: \n", str(train_loader))
    
    for i, (img, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)

        print(f"img ********************************* : {img.shape}")
        with torch.no_grad():
            teacher_output, teacher_kd_list, teacher_resize_list = teacher_model(img)

        student_output, student_kd_list, student_resize_list = student_model(img)       

        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        
        print(f"target ******************************: {target.shape}")
        print("target = ", target.sum().item())
        loss = criterion(student_kd_list, teacher_kd_list, student_resize_list, teacher_resize_list, student_output, teacher_output, target)
        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('\n\n\nEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses ))
            print("accuracy: ", student_output.sum()/target.sum()*100, " % \n\n")
    
def validate(val_list, model, criterion):
    print ('\nbegin test')

    # test_loader = torch.utils.data.DataLoader(
    # dataset.listDataset(val_list,
    #                shuffle=False,
    #                transform=transforms.Compose([
    #                    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225]),
    #                ]),  train=False),
    # batch_size=args.batch_size)    

    _test_loader = dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], # from img net 
                                                                             std=[0.229, 0.224, 0.225]),
                                                        ]), 
                            ) 
    test_loader = DataLoader( _test_loader, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)    
 

    model.eval()
    mae = 0
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output, _, _ = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())

    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
