# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2017

@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import os
import math
import random
import numpy as np
from PIL import Image

class ifs_function():
    def __init__(self, prev_x, prev_y, save_root,fractal_name,fractal_weight_count):
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
        # Root path for image save
        self.save_root = save_root
        # Fractal pattern's name
        self.fractal_name = fractal_name
        # Number of fractal pattern
        self.fractal_weight_count = fractal_weight_count

    def set_param(self,a,b,c,d,e,f,proba,**kwargs):
        # Calculate parameter set and select function
        if "weight_a" in kwargs:
            a *= kwargs["weight_a"]
        if "weight_b" in kwargs:
            b *= kwargs["weight_b"]
        if "weight_c" in kwargs:
            c *= kwargs["weight_c"]
        if "weight_d" in kwargs:
            d *= kwargs["weight_d"]
        if "weight_e" in kwargs:
            e *= kwargs["weight_e"]
        if "weight_f" in kwargs:
            f *= kwargs["weight_f"]
        temp_function  = {"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"proba":proba} 
        self.function.append(temp_function)
        # Plus probability when function is added
        self.temp_proba += proba
        self.select_function.append(self.temp_proba)

    def calculate(self,iteration):
        # Fix random seed
        np.random.seed(100)
        rand = np.random.random(iteration)
        select_function = self.select_function
        function = self.function
        prev_x,prev_y = self.prev_x, self.prev_y
        for i in range(iteration-1):
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
            print("x is nan")
            nan_index = np.where(np.isnan(xs))
            extend = np.array(range(nan_index[0][0]-100,nan_index[0][0]))
            delete_row = np.append(extend,nan_index)
            xs = np.delete(xs,delete_row,axis=0)
            ys = np.delete(ys,delete_row,axis=0)
            print ("early_stop: %d" % len(xs))
        if np.any(np.isnan(ys)):
            print("y is nan")
            nan_index = np.where(np.isnan(ys))
            extend = np.array(range(nan_index[0][0]-100,nan_index[0][0]))
            delete_row = np.append(extend,nan_index)
            xs = np.delete(xs,delete_row,axis=0)
            ys = np.delete(ys,delete_row,axis=0)
            print ("early_stop: %d" % len(ys))
            
        if np.min(xs) < 0.0:
            xs -= np.min(xs)
        if np.min(ys) < 0.0:
            ys -= np.min(ys)
        xmax,xmin,ymax,ymin = np.max(xs),np.min(xs),np.max(ys),np.min(ys)
        self.xs = np.uint16(xs / (xmax-xmin) * float(image_x-2*pad_x)+float(pad_x))    
        self.ys = np.uint16(ys / (ymax-ymin) * float(image_y-2*pad_y)+float(pad_y))

    def __transpose(self,ori_image,trans_type):
        if trans_type == 0:
            pass
        elif trans_type == 1:
            ori_image = ori_image.transpose(Image.FLIP_TOP_BOTTOM)
        elif trans_type == 2:
            ori_image = ori_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif trans_type == 3:
            ori_image = ori_image.transpose(Image.FLIP_TOP_BOTTOM)
            ori_image = ori_image.transpose(Image.FLIP_LEFT_RIGHT)
        return ori_image

    def draw_point(self,image_x,image_y,pad_x,pad_y,set_color,count):
        self.__rescale(image_x,image_y,pad_x,pad_y)
        image = np.array(Image.new("RGB", (image_x, image_y)))
        for i in range(len(self.xs)):
            if set_color == "color":
                image[self.ys[i],self.xs[i],:] = self.convert_color(i,128)
            else:
                image[self.ys[i],self.xs[i],:] = 127,127,127
        image = Image.fromarray(image)

        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        for trans_type in range(4):
            trans_image = self.__transpose(image,trans_type)
            trans_image.save(os.path.join(self.save_root, self.fractal_name, self.fractal_name + "_" + self.fractal_weight_count + "_count_" + str(count) + "_flip" + str(trans_type) + ".png"))
            #trans_image.close()
        image.close()

    def draw_patch(self,image_x,image_y,pad_x,pad_y,set_color,count):
        self.__rescale(image_x,image_y,pad_x,pad_y)
        image = Image.new("RGB", (image_x, image_y))
        #mask_pattern = '{:09b}'.format(random.randrange(1,512))

        for i in range(len(self.xs)):
            mask_pattern = '{:09b}'.format(random.randrange(1,512))
            if set_color == "color":
                patch = self.make_patch3_3(mask_pattern,[self.convert_color(i,128)])
            else:
                patch = self.make_patch3_3(mask_pattern,[127,127,127])
            image.paste(patch, (self.xs[i]+1, self.ys[i]+1))
        patch.close()
        # Coordinate transformation
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        for trans_type in range(4):
            trans_image = self.__transpose(image,trans_type)
            trans_image.save(os.path.join(self.save_root, self.fractal_name, self.fractal_name + "_" + self.fractal_weight_count + "_count_" + str(count) + "_flip" + str(trans_type) + ".png"))
        trans_image.close()
        image.close()

    def convert_color(self,color,n_div):
        S = 0.75
        H = 2.0*math.pi * (float(color % n_div)/float(n_div))
        if H >= 2.0*math.pi:
            H -= 2.0*math.pi
        h = math.floor((3.0/math.pi)*H)
        I = 0.75
        P = I*(1.0-S)
        Q = I*(1.0-S*((3.0/math.pi)*H-h))
        T = I*(1.0-S*((1.0-(3.0/math.pi)*H)+h))
        I = np.uint8(I*255.0)
        P = np.uint8(P*255.0)
        Q = np.uint8(Q*255.0)
        T = np.uint8(T*255.0)
        # Ordered by BGR
        if h == 0.0:
            return (P,T,I)
        elif h == 1.0:
            return (P,I,Q)
        elif h == 2.0:
            return (T,I,P)
        elif h == 3.0:
            return (I,Q,P)
        elif h == 4.0:
            return (I,P,T)
        elif h == 5.0:
            return (Q,P,I)

    def make_patch3_3(self,mask,patch_color):
        patch_color = np.array(patch_color)
        patch = np.zeros((3,3,3),np.uint8)
        for i in range(0,3):
            for j in range(0,3):
                patch[i,j,:] = patch_color*int(mask[i*3+j])
        return Image.fromarray(patch)
