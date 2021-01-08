# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import os
import sys

import torch
import torch.nn as nn

from densenet import *
from resnet import *
from resnext import *
#from bn_alexnet import bn_alexnet, bn_alex_deepclustering,rot_AlexNet,load_pretrained
from bn_alexnet import bn_alexnet
from vgg import vgg16_bn, vgg19_bn

def model_select(args):
	
	MODEL_ROOT = args.path2weight

	# Batch Normalized AlexNet
	if args.usenet == "bn_alexnet":
		model = bn_alexnet(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		last_layer = nn.Sequential(nn.Linear(4096, args.numof_classes))
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			model = bn_alexnet(num_classes=1000)
			weight_name = os.path.join(MODEL_ROOT, "bn_alexnet-imagenet.pth.tar")
			assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
			print ("use imagenet pretrained model")
			checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			model.load_state_dict(state_dict)
			model.classifier = nn.Sequential(*(list(model.classifier.children())[:-1]+list(last_layer)))
		removed = list(model.classifier.children())[:-1]
		model.classifier = torch.nn.Sequential(*removed)
		model.classifier = torch.nn.Sequential(model.classifier, nn.Linear(4096, args.numof_classes))

	# ResNet-18
	if args.usenet == "resnet18":
		last_layer = nn.Linear(512, args.numof_classes)
		model = resnet18(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name, map_location=lambda storage, loc: storage)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet18(pretrained=True)
		model.fc = last_layer

	# ResNet-34
	if args.usenet == "resnet34":
		last_layer = nn.Linear(512, args.numof_classes)
		model = resnet34(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name, map_location=lambda storage, loc: storage)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet34(pretrained=True)
		model.fc = last_layer

	# ResNet-50
	if args.usenet == "resnet50":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnet50(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet50(pretrained=True)
		model.fc = last_layer

	# ResNet-101
	if args.usenet == "resnet101":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnet101(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet101(pretrained=True)
		model.fc = last_layer

	# ResNet-152
	if args.usenet == "resnet152":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnet152(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = resnet152(pretrained=True)
		model.fc = last_layer

	# ResNet-200
	if args.usenet == "resnet200":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnet200(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		model.fc = last_layer

	# ResNeXt-101
	if args.usenet == "resnext101":
		last_layer = nn.Linear(2048, args.numof_classes)
		model = resnext101(num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			model = resnext101(pretrained=False, num_classes=1000)
			weight_name = os.path.join(MODEL_ROOT, "resnext101_imagenet.pth.tar")
			assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
			print ("use imagenet pretrained model")
			checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			model.load_state_dict(state_dict)
		model.fc = last_layer

	# DenseNet-161
	if args.usenet == "densenet161":
		last_layer = nn.Linear(2208, args.numof_classes)
		model = densenet161(pretrained=False, num_classes=args.numof_pretrained_classes)
		weight_name = os.path.join(args.path2weight, args.dataset + "_" + args.usenet + "_epoch" + str(args.useepoch) + ".pth")
		
		# FractalDB pre-trained model
		if os.path.exists(weight_name):
			print ("use pretrained model : %s" % weight_name)
			param = torch.load(weight_name)
			model.load_state_dict(param)
		# ImageNet pre-trained model
		elif args.dataset == "imagenet":
			print ("use imagenet pretrained model")
			model = densenet161(pretrained=True)
		model.classifier = last_layer

	return model
