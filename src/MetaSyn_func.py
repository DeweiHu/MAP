# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:35:17 2023

@author: hudew
"""

import sys
sys.path.insert(0, "E:\\MetaSyn\\src\\")
sys.path.insert(0,'E:\\tools\\')

import util
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import pickle


def dir_mixup(im_1, im_2, im_3, alpha):        
    theta = np.random.dirichlet(alpha)
    opt = theta[0]*util.ImageRescale(im_1,[0,1]) + \
          theta[1]*util.ImageRescale(im_2,[0,1]) + \
          theta[2]*util.ImageRescale(im_3,[0,1])
    return opt


def data_split(meta_test):
    meta_train = list(np.arange(1,5))
    meta_train.remove(meta_test)
    
    for i in range(len(meta_train)):
        meta_train[i] = "augments_{}".format(meta_train[i])
    meta_test = ["augments_{}".format(meta_test)]
    
    return meta_train, meta_test


class get_DirMixup_dataset(Data.Dataset):
    def __init__(self, mtrain_data, mtrain_gt, n_mixup, n_patch, patch_size, 
                 alpha=(1.5,1.5,1.5)):
        # hyper-parameters
        self.n_mixup = n_mixup
        self.n_patch = n_patch
        self.patch_size = patch_size
        self.alpha = alpha
        
        # output
        self.x_list = []
        self.y_list = []
        
        keys = list(mtrain_data)
        assert len(keys) == 3, "sample space error"
        
        for i in range(len(mtrain_data[keys[0]])):
            im_1 = mtrain_data[keys[0]][i]
            im_2 = mtrain_data[keys[1]][i]
            im_3 = mtrain_data[keys[2]][i]
            y = mtrain_gt[i]
            
            mixup_list = self.get_mixup_sample(im_1, im_2, im_3)
            pair_data = self.get_crop_data(mixup_list, y)
        
            self.x_list += pair_data["x_stack"]
            self.y_list += pair_data["y"]
            
    def __len__(self,):
        return len(self.x_list)

    
    def __getitem__(self, idx):    
        x = self.x_list[idx]
        y = self.y_list[idx]
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.int64)
        return x_tensor, y_tensor
    
    
    def get_mixup_sample(self, im_1, im_2, im_3):
        mixup_list = []
        for i in range(self.n_mixup):
            mixup_list.append(dir_mixup(im_1, im_2, im_3, self.alpha))
        return mixup_list


    def get_sample_coordinate(self, h, w):
        sample_x = np.random.randint(0, h-self.patch_size[0], self.n_patch)
        sample_y = np.random.randint(0, w-self.patch_size[1], self.n_patch)
        return sample_x, sample_y
    
    
    def get_crop_data(self, im_list, gt):
        pair_data = {"x_stack":[], "y":[]}
        h,w = im_list[0].shape
        crd_x, crd_y = self.get_sample_coordinate(h, w)
        
        for i in range(self.n_patch):
            x = []
            y = gt[crd_x[i]:crd_x[i]+self.patch_size[0],
                   crd_y[i]:crd_y[i]+self.patch_size[1]]
            for j in range(len(im_list)):
                x.append(im_list[j][crd_x[i]:crd_x[i]+self.patch_size[0],
                                 crd_y[i]:crd_y[i]+self.patch_size[1]])
            
            pair_data["x_stack"].append(np.array(x))
            pair_data["y"].append(y)
    
        return pair_data
            
def load_DirMixup_data(mtrain_data, mtrain_gt, n_mixup, n_patch, patch_size, batch_size):
    dataset = get_DirMixup_dataset(mtrain_data, mtrain_gt, n_mixup, n_patch, patch_size)
    loader = Data.DataLoader(dataset, batch_size, shuffle=True)        
    return loader


class SampleMatrix(nn.Module):
    def __init__(self,):
        super(SampleMatrix, self).__init__()
        self.softmax = nn.Softmax2d()
        
    def forward(self, tensor, t_type, idx=0):
        if t_type == "pred":
            pred_tensor = torch.argmax(self.softmax(tensor),dim=1)
            matrix = pred_tensor[idx,:,:].detach().cpu().numpy()
        elif t_type == "latent":
            matrix = tensor[idx,0,:,:].detach().cpu().numpy()
        elif t_type == "gt":
            matrix = tensor[idx,:,:].detach().cpu().numpy()
        else:
            raise ValueError
        
        return matrix
    

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_inner = checkpoint['lr_sched']

    return model, optimizer, epoch, scheduler_inner
