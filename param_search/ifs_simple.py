# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2017
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import math
import random
import numpy as np
from PIL import Image

class ifs_function():
	def __init__(self, prev_x, prev_y):
		# previous (x, y)
		self.prev_x,self.prev_y = prev_x,prev_y
		# IFS function
		self.function  = []
		# Iterative results
		self.xs,self.ys = [],[]
		# Add initial value
		self.xs.append(prev_x),self.ys.append(prev_y)
		# Select function
		self.select_function = []
		# Calculate select function
		self.temp_proba = 0.0

	def set_param(self,a,b,c,d,e,f,proba,**kwargs):
		# Initial parameters & select function
		temp_function  = {"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"proba":proba} 
		self.function.append(temp_function)
		# Plus probability when function is added
		self.temp_proba += proba
		self.select_function.append(self.temp_proba)

	def calculate(self,iteration):
		# Fix random seed
		rand = np.random.random(iteration)
		select_function = self.select_function
		function = self.function
		prev_x,prev_y = self.prev_x, self.prev_y
		#for i in xrange(iteration-1): #python2.x
		for i in range(iteration-1):   #python3.x
			for j in range(len(select_function)):
				if rand[i] <= select_function[j]:
					next_x = prev_x * function[j]["a"] + prev_y * function[j]["b"] + function[j]["e"]
					next_y = prev_x * function[j]["c"] + prev_y * function[j]["d"] + function[j]["f"]
					break
			self.xs.append(next_x),self.ys.append(next_y)
			prev_x = next_x
			prev_y = next_y

	# Inner function
	def __rescale(self,image_x,image_y,pad_x,pad_y):
		# Scale adjustment
		xs = np.array(self.xs)
		ys = np.array(self.ys)
		if np.any(np.isnan(xs)):
			#print("x is nan")
			nan_index = np.where(np.isnan(xs))
			extend = np.array(range(nan_index[0][0]-100,nan_index[0][0]))
			delete_row = np.append(extend,nan_index)
			xs = np.delete(xs,delete_row,axis=0)
			ys = np.delete(ys,delete_row,axis=0)
			#print ("early_stop: %d" % len(xs))
		if np.any(np.isnan(ys)):
			#print("y is nan")
			nan_index = np.where(np.isnan(ys))
			extend = np.array(range(nan_index[0][0]-100,nan_index[0][0]))
			delete_row = np.append(extend,nan_index)
			xs = np.delete(xs,delete_row,axis=0)
			ys = np.delete(ys,delete_row,axis=0)
			#print ("early_stop: %d" % len(ys))
		if np.min(xs) < 0.0:
			xs -= np.min(xs)
		if np.min(ys) < 0.0:
			ys -= np.min(ys)
		xmax,xmin,ymax,ymin = np.max(xs),np.min(xs),np.max(ys),np.min(ys)
		self.xs = np.uint16(xs / (xmax-xmin) * float(image_x-2*pad_x)+float(pad_x))	
		self.ys = np.uint16(ys / (ymax-ymin) * float(image_y-2*pad_y)+float(pad_y))

	def draw_point(self,image_x,image_y,pad_x,pad_y,set_color,count):
		self.__rescale(image_x,image_y,pad_x,pad_y)
		image = np.array(Image.new("RGB", (image_x, image_y)))
		#for i in xrange(len(self.xs)): #python2.x
		for i in range(len(self.xs)):   #python3.x
			image[self.ys[i],self.xs[i],:] =127,127,127
		return image
