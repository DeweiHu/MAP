# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:14:42 2023

@author: hudew
"""

import sys
sys.path.insert(0,"E:\\tools\\")
import util

import numpy as np
import imageio
import pickle
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm

data_root = {}
data = {}

data_root["drive"] = "E:\\DRIVE\\training\\"
data_root["stare"] = "E:\\STARE\\"
data_root["aria_control"] = "E:\\ARIA\\control\\"
data_root["aria_diabetic"] = "E:\\ARIA\\diabetic\\"
data_root["aria_amd"] = "E:\\ARIA\\amd\\"
data_root["hrf_control"] = "E:\\HRF\\control\\"
data_root["hrf_diabetic"] = "E:\\HRF\\diabetic\\"
data_root["hrf_glaucoma"] = "E:\\HRF\\glaucoma\\"
data_root["chase"] = "E:\\CHASE_DB1\\"
data_root["prime_fa"] = "E:\\PRIME-FP20\\prime_fa\\"
data_root["prime_fundus"] = "E:\\PRIME-FP20\\prime_fundus\\"
data_root["recovery_fa"] = "E:\\FA\\RECOVERY-FA19\\"
data_root["rose"] = "E:\\ROSE\\data\\ROSE-1\\SVC\\train\\"
data_root["octa500"] = "E:\\OCTA500\\GT\\OCTA-500\\OCTA_6M\\"


class pair:
    '''
    img: normalize all input image to range [0,1], type = np.float32
     gt: binarized to 0,1, type = np.uint8
    convert color image to grayscale by taking the green channel.
    im_dir: list of directories of each image
    gt_dir: list of directories of each label 
    '''
    def __init__(self, im_dir, gt_dir, msk_dir=None):
        self.img_dir = im_dir
        self.gt_dir = gt_dir
        self.msk_dir = msk_dir
        
    def reader(self, root):
        fmt = root[-3:]
        if fmt == "gif":
            im = imageio.imread(root)
        else:
            im = np.array(Image.open(root))
        return im
    
    def normalize(self, im, im_type):
        if im_type == "img":
            if len(im.shape) == 3:
                opt = util.ImageRescale(np.float32(im[:,:,1]), [0,1])
            else:
                opt = util.ImageRescale(np.float32(im), [0,1])
        
        elif im_type == "gt":
            # remove other labels (e.g. fovea)
            opt = np.uint8(im == im.max())
            
        elif im_type == "mask":
            opt = np.uint8(im == im.max())
        
        else:
            raise ValueError
            
        return opt
    
    def __call__(self,):
        img_list = []
        gt_list = []
        mask_list = []
        
        for i in range(len(self.img_dir)):
            im = self.reader(self.img_dir[i])
            gt = self.reader(self.gt_dir[i])
            
            if self.msk_dir == None:
                mask = np.ones([im.shape[0],im.shape[1]], dtype=np.uint8)
            else:
                mask = self.reader(self.msk_dir[i])
                if len(mask.shape) == 3:
                    mask = np.uint8(mask == mask.max())
                    mask = mask[:,:,0]
                    
            img_list.append(self.normalize(im, "img"))
            gt_list.append(self.normalize(gt, "gt"))
            mask_list.append(self.normalize(mask, "mask"))
            
        return img_list, gt_list, mask_list
    
#%%
for key in list(data_root):
        
    img_root = data_root[key] + "img\\"
    gt_root = data_root[key] + "gt\\"
    msk_root = data_root[key] + "mask\\"
    
    img_dirs = []
    gt_dirs = []
    msk_dirs = []
        
    if key.startswith("aria"):
        for file in os.listdir(img_root):
            if file.endswith(".tif"):
                img_dirs.append(img_root + file)
                
                name,_ = file.split(".")
                name = name.replace(" ","")
                gt_dirs.append(gt_root + name + "_BDP.tif")
    else:
        for file in os.listdir(img_root):
            img_dirs.append(img_root + file)
        for file in os.listdir(gt_root):
            gt_dirs.append(gt_root + file)
    
    if os.path.exists(msk_root):
        for file in os.listdir(msk_root):
            msk_dirs.append(msk_root + file)
    else:
        msk_dirs = None

    dr = pair(img_dirs, gt_dirs, msk_dirs)
    data[key+"_im"], data[key+"_gt"], data[key+"_msk"] = dr()

    print("{} loaded.".format(key))

#%%    
save_root = "E:\\MetaSyn\\data\\"
 
with open(save_root + "data.pickle", "wb") as handle:
    pickle.dump(data, handle)
    
    
    