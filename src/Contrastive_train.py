# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:13:43 2023

@author: hudew
"""

import sys 
sys.path.insert(0, "E:\\MetaSyn\\src\\")
sys.path.insert(0, "E:\\tools\\")

import MetaSyn_func as func
import models
import loss
import util

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
 
model_root = "E:\\Model\\"
data_root = "E:\\MetaSyn\\data\\"
check_root = "E:\\MetaSyn\\checkpoint\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% load dataset
mtrain = 4
meta_test_list, meta_train_list = func.data_split(mtrain) 

mtrain_data = {}
for key in meta_train_list:
    with open(data_root + key + ".pickle", "rb") as handle:
        mtrain_data[key] = pickle.load(handle)

mtest_data = {}
for key in meta_test_list:
    with open(data_root + key + ".pickle", "rb") as handle:
        mtest_data[key] = pickle.load(handle)

with open(data_root + "gt.pickle", "rb") as handle:
    gt = pickle.load(handle)

with open("E:\\static representation\\data\\" + "raw_data.pickle","rb") as handle:
    validation_data = pickle.load(handle)
    
validation_set = ["drive","octa500","rose","aria_control","aria_amd","aria_diabetic"]

n_epoch = 50
n_mixup = 5
n_patch = 40
patch_size = [256,256]
batch_size = 1

#%% setup model
model_anchor = models.res_UNet([8,32,32,64,64,16], 1, 2).to(device)
model = models.res_UNet([8,32,32,64,64,16], 1, 2).to(device)
#model.load_state_dict(torch.load(model_root + "rUNet.pt"))

optimizer_inner = torch.optim.Adam(model.parameters(),lr=1e-3)
scheduler_inner = StepLR(optimizer_inner, step_size=5, gamma=0.6)
optimizer_outer = torch.optim.Adam(model.parameters(),lr=1e-3)
scheduler_outer = StepLR(optimizer_outer, step_size=5, gamma=0.6)

# losses
DSC_loss = loss.DiceBCELoss()
CE_loss = nn.CrossEntropyLoss()
L1_loss = nn.L1Loss()

#%% training

get_sample = func.SampleMatrix()

for epoch in range(n_epoch):
    # load data
    alpha = tuple([random.randint(8,10),
                   random.randint(5,10),
                   random.randint(1,5)])
    train_loader = func.load_DirMixup_data(mtrain_data, mtest_data, gt, n_mixup, 
                                       n_patch, patch_size, alpha, batch_size)
    # inner loop
    values = range(len(train_loader))
    with tqdm(total=len(values)) as pbar:
        
        for step, (x,y,anchor) in enumerate(train_loader):
            x = Variable(anchor).to(device)
            y = Variable(y).to(device)
            
            model.train()
            y_seg, _ = model(x)
            pred = torch.argmax(get_sample.softmax(y_seg), dim=1)
            
            seg_loss = CE_loss(y_seg, y) + DSC_loss(pred, y)
            optimizer_inner.zero_grad()
            seg_loss.backward()
            optimizer_inner.step()
            
            pbar.update(1)
            pbar.set_description('InnerLoop: %d. inner-loss: %.4f.' \
                                 %(epoch+1, seg_loss.item()))
    
    torch.save(model.state_dict(),check_root + "model_anchor.pt")
    model_anchor.load_state_dict(torch.load(check_root + "model_anchor.pt"))
    
    # outer loop    
    with tqdm(total=len(values)) as pbar:
        
        for step, (x,y,anchor) in enumerate(train_loader):
            x = Variable(torch.swapaxes(x,0,1)).to(device)
            b,c,h,w = x.size()
            y = Variable(y.repeat(b,1,1)).to(device)
            anchor = Variable(anchor).to(device)
            
            model_anchor.eval()
            model.train()
            
            y_seg_anchor, cats_anchor = model_anchor(anchor)
            pred_anchor = torch.argmax(get_sample.softmax(y_seg_anchor), dim=1)
            y_seg, cats = model(x)
            pred = torch.argmax(get_sample.softmax(y_seg), dim=1)
            
            seg_loss = CE_loss(y_seg, y) + DSC_loss(pred, y)
            
            sim_loss = torch.tensor(0).type(torch.FloatTensor).to(device)
            for k in range(len(cats)-1,len(cats)):
                z_anchor = cats_anchor[k].repeat(b,1,1,1)
                z = cats[k]
                sim_loss += L1_loss(z, z_anchor)    
#            
#            z_anchor = cats_anchor[-1].repeat(b,1,1,1)
#            z = cats[-1]
#            sim_loss = L1_loss(z, z_anchor)
                
            losses = seg_loss + sim_loss
            
            optimizer_outer.zero_grad()
            losses.backward()
            optimizer_outer.step()
            
            with torch.no_grad():
                gt_mat = get_sample(y, "gt")
                pred_mat = get_sample(y_seg, "pred")
                dsc_seg = util.dice(pred_mat, gt_mat)
        
                pbar.update(1)
                pbar.set_description('OuterLoop: %d. seg-loss: %.4f. sim-loss: %.4f. Dice: %.4f.' \
                                     %(epoch+1, seg_loss.item(), sim_loss.item(), dsc_seg))
                
                if step % (len(train_loader)-1) == 0 and step != 0:
                    func.display_array(x[:n_mixup,:,:,:], anchor, y_seg[:n_mixup,:,:,:], y_seg_anchor, y)
                    func.display_array(x[n_mixup:,:,:,:], anchor, y_seg[n_mixup:,:,:,:], y_seg_anchor, y)
        
    # validation
    print("Validation with α = ({},{},{}): \n----------------------------"
          .format(alpha[0],alpha[1],alpha[2]))
    for item in validation_set:
        valid_im_list = validation_data[item + "_im"]
        valid_gt_list = validation_data[item + "_gt"]
        vresult = []
    
        for i in range(len(valid_im_list)):
            vx = valid_im_list[i]
            vy = valid_gt_list[i]
            
            if item.startswith("aria") or item.startswith("drive"):
                vx = func.CLAHE(vx.max()-vx, 5)
    
            test = func.Size_Adaptive_Test(vx)
            pred = test.model_test(model, device)
            vresult.append(util.dice(pred,vy)) 
            
        print("{} meam DICE: {}".format(item, np.array(vresult).mean()))
    
    print("----------------------------")
    
    scheduler_inner.step()
    scheduler_outer.step()
    
#name = 'MetaSyn_{}.pt'.format(mtrain)
#torch.save(model.state_dict(),model_root+name)

#%%
#print("Validation with α = ({},{},{}): \n----------------------------"
#          .format(alpha[0],alpha[1],alpha[2]))
#for item in validation_set:
#    valid_im_list = validation_data[item + "_im"]
#    valid_gt_list = validation_data[item + "_gt"]
#    vresult = []
#
#    for i in range(len(valid_im_list)):
#        vx = valid_im_list[i]
#        vy = valid_gt_list[i]
#        
#        if item.startswith("aria") or item.startswith("drive"):
#            vx = func.CLAHE(vx.max()-vx, 5)
#
#        test = func.Size_Adaptive_Test(vx)
#        pred = test.model_test(model, device)
#        vresult.append(util.dice(pred,vy)) 
#        
#    print("{} meam DICE: {}".format(item, np.array(vresult).mean()))
#
#print("----------------------------")