# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:20:35 2023

@author: hudew
"""

import sys 
sys.path.insert(0, "E:\\MetaSyn\\src\\")
sys.path.insert(0, "E:\\tools\\")

import MetaSyn_func as func
import models
import util

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable


model_root = "E:\\Model\\"
data_root = "E:\\MetaSyn\\data\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.res_UNet([8,32,32,64,64,16], 1, 2).to(device)
model.load_state_dict(torch.load(model_root + "rUNet.pt"))

with open(data_root+"data.pickle", "rb") as handle:
    data = pickle.load(handle)

#%% load data
test_dataset = "prime_fundus"
im_list = data[test_dataset + "_im"]
msk_list = data[test_dataset + "_msk"]
gt_list = data[test_dataset + "_gt"]

#%% test
test_method = "patch"
patch_size = [512,512]
stride = 200
dice_list = []

values = range(len(im_list))

with tqdm(total=len(values)) as pbar:
    for i in range(len(im_list)):
        if test_dataset.startswith("octa500") or test_dataset.startswith("rose") \
        or test_dataset.startswith("recovery_fa") or test_dataset.startswith("prime_fa"):
            im = im_list[i]
        else:
            im = func.reverse_int(im_list[i])
        
        gt = gt_list[i]
        mask = msk_list[i]
        
        if test_method == "patch":
            test = func.Patch_Pad_Test(im, patch_size, stride)
            pred = test.model_test(model, device)
        elif test_method == "full":
            test = func.Size_Adaptive_Test(im)
            pred = test.model_test(model, device)
        else:
            raise ValueError
        
        dice_list.append(util.dice(pred * mask, gt))
        pbar.update(1)
        
print(np.array(dice_list).mean())