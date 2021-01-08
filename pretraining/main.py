# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import datetime
import os
import random
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from args import conf
from alex import bn_alexnet
from vgg import vgg16_bn, vgg19_bn
from resnet import *
from resnext import *
from densenet  import * 
from train_val import train, validate

def worker_init_fn(worker_id):
    random.seed(worker_id)

def model_select(args):
    if args.usenet == "bn_alexnet":
        model = bn_alexnet(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg16":
        model = vgg16_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg19":
        model = vgg19_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet18":
        model = resnet18(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet34":
        model = resnet34(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet50":
        model = resnet50(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet101":
        model = resnet101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet152":
        model = resnet152(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet200":
        model = resnet200(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnext101":
        model = resnext101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "densenet161":
        model = densenet161(pretrained=False, num_classes=args.numof_classes).to(device)
        return model


if __name__== "__main__":

    # Processing time
    starttime = time.time()
    
    # Option
    args = conf()
    print(args)

    # GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #to deterministic
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Training settings
    normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([transforms.RandomCrop((args.crop_size,args.crop_size)),
                                        transforms.ToTensor(), normalize])
    train_dataset = datasets.ImageFolder(args.path2traindb, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    
    # Basically, the FractalDB pre-training doesn't require validation phase
    if args.val:
        val_transform = transforms.Compose([transforms.Resize((args.crop_size,args.crop_size), interpolation=2),
                                         transforms.ToTensor(), normalize])
        val_dataset = datasets.ImageFolder(args.path2valdb, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    # Model & optimizer
    model = model_select(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    if not args.no_multigpu:
        model = nn.DataParallel(model)
    
    # FractalDB Pre-training
    iteration = (args.start_epoch-1)*len(train_loader)
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        scheduler.step()
        iteration += len(train_loader)
        if args.val:
            validation_loss = validate(args, model, device, val_loader, criterion, iteration)
        if epoch % args.save_interval == 0:
            if args.no_multigpu:
                model_state = model.cpu().state_dict()
            else:
                model_state = model.module.cpu().state_dict()            
            saved_weight = "{}/{}_{}_epoch{}.pth.tar".format(args.path2weight, args.dataset, args.usenet, epoch)
            torch.save(model_state, saved_weight.replace('.tar',''))
            checkpoint = "{}/{}_{}_checkpoint.pth.tar".format(args.path2weight, args.dataset, args.usenet)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_state,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),}, checkpoint)
            model = model.to(device)
    torch.save(model_state, saved_weight.replace('.tar',''))

    # Processing time
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval/3600), int((interval%3600)/60), int((interval%3600)%60)))