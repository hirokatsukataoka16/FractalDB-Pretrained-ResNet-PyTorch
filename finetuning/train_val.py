# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import sys
import numpy as np

import torch
import torch.nn as nn

# Training
def train(args, model, device, train_loader, optimizer, epoch, iteration):
	model.train()
	criterion = nn.CrossEntropyLoss(size_average=True) # previous PyTorch ver.
	#criterion = nn.CrossEntropyLoss(reduction='sum')
	for i_batch, sample_batched in enumerate(train_loader):
		data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
		optimizer.zero_grad()
		output = model(data)
		pred = output.max(1, keepdim=True)[1]
		correct = pred.eq(target.view_as(pred)).sum().item()
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if i_batch % args.log_interval == 0:
			sys.stdout.write("\repoch:{0:>3} iteration:{1:>6} train_loss: {2:.6f} train_accracy: {3:5.2f}%".format(
							epoch, iteration, loss.item(), 100.*correct/float(len(sample_batched["label"]))))
			sys.stdout.flush()
		iteration += 1

# Validation
def val(args, model, device, test_loader, iteration):
	model.eval()
	criterion = nn.CrossEntropyLoss(size_average=False) # previous PyTorch ver.
	#criterion = nn.CrossEntropyLoss(reduction='sum')
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(test_loader):
			data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
			output = model(data)
			test_loss += criterion(output, target).item()
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= float(len(test_loader.dataset))
	correct /= float(len(test_loader.dataset))
	print("\nValidation: Accuracy: {0:.2f}%  test_loss: {1:.6f}".format(100. * correct, test_loss))
	return test_loss, 100. * correct
