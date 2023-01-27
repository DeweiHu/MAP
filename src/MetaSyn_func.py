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
from torch.autograd import Variable

import pickle
import matplotlib.pyplot as plt



def dir_mixup(im_1, im_2, im_3, alpha):        
    theta = np.random.dirichlet(alpha)
    opt = theta[0]*util.ImageRescale(im_1,[0,1]) + \
          theta[1]*util.ImageRescale(im_2,[0,1]) + \
          theta[2]*util.ImageRescale(im_3,[0,1])
    return util.ImageRescale(opt,[0,1])


def data_split(meta_test):
    meta_train = list(np.arange(1,5))
    meta_train.remove(meta_test)
    
    for i in range(len(meta_train)):
        meta_train[i] = "augments_{}".format(meta_train[i])
    meta_test = ["augments_{}".format(meta_test)]
    
    return meta_train, meta_test


class get_DirMixup_dataset(Data.Dataset):
    def __init__(self, mtrain_data, mtest_data, gt, n_mixup, n_patch, patch_size, 
                 alpha=(1.5,1.5,1.5)):
        # hyper-parameters
        self.n_mixup = n_mixup
        self.n_patch = n_patch
        self.patch_size = patch_size
        self.alpha = alpha
        
        # output
        self.x_list = []
        self.y_list = []
        self.anchor_list = []
        
        trainkey = list(mtrain_data)[0]
        
        testkeys = list(mtest_data)
        assert len(testkeys) == 3, "sample space error"
        
        for i in range(len(mtest_data[testkeys[0]])):
            im_1 = mtest_data[testkeys[0]][i]
            im_2 = mtest_data[testkeys[1]][i]
            im_3 = mtest_data[testkeys[2]][i]
            y = gt[i]
            
#            anchor = 0.4*mtrain_data[trainkey][i] + 0.6*np.float32(y)
            anchor = util.ImageRescale(mtrain_data[trainkey][i],[0,1])
            
            mixup_list = self.get_mixup_sample(im_1, im_2, im_3)
            pair_data = self.get_crop_data(mixup_list, y, anchor)
        
            self.x_list += pair_data["x_stack"]
            self.y_list += pair_data["y"]
            self.anchor_list += pair_data["anchor"]
            
    def __len__(self,):
        return len(self.x_list)

    
    def __getitem__(self, idx):    
        x = self.x_list[idx]
        y = self.y_list[idx]
        im_anchor = self.anchor_list[idx]
        
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.int64)
        anchor_tensor = torch.tensor(im_anchor).type(torch.FloatTensor)
        
        return x_tensor, y_tensor, anchor_tensor
    
    
    def get_mixup_sample(self, im_1, im_2, im_3):
        mixup_list = []
        for i in range(self.n_mixup):
            mixup_list.append(dir_mixup(im_1, im_2, im_3, self.alpha))
        return mixup_list


    def get_sample_coordinate(self, h, w):
        sample_x = np.random.randint(0, h-self.patch_size[0], self.n_patch)
        sample_y = np.random.randint(0, w-self.patch_size[1], self.n_patch)
        return sample_x, sample_y
    
    
    def get_crop_data(self, im_list, gt, anchor):
        pair_data = {"x_stack":[], "y":[], "anchor":[]}
        h,w = im_list[0].shape
        crd_x, crd_y = self.get_sample_coordinate(h, w)
        
        for i in range(self.n_patch):
            x = []
            y = gt[crd_x[i]:crd_x[i]+self.patch_size[0],
                   crd_y[i]:crd_y[i]+self.patch_size[1]]
            im_anchor = anchor[crd_x[i]:crd_x[i]+self.patch_size[0],
                               crd_y[i]:crd_y[i]+self.patch_size[1]]
            for j in range(len(im_list)):
                x.append(im_list[j][crd_x[i]:crd_x[i]+self.patch_size[0],
                                 crd_y[i]:crd_y[i]+self.patch_size[1]])
            
            pair_data["x_stack"].append(np.array(x))
            pair_data["y"].append(y)
            pair_data["anchor"].append(im_anchor[None,:,:])
    
        return pair_data
            
def load_DirMixup_data(mtrain_data, mtest_data, gt, n_mixup, n_patch, patch_size, alpha, batch_size):
    dataset = get_DirMixup_dataset(mtrain_data, mtest_data, gt, n_mixup, n_patch, patch_size, alpha)
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


def display_array(x, anchor, pred, pred_anchor, gt):
    b,c,h,w = x.size()
    get_sample = SampleMatrix()
    
    anchor_mat = get_sample(anchor, "latent")
    pred_anchor_mat = get_sample(pred_anchor, "pred")
    gt_mat = get_sample(gt, "gt")
    pred_anchor_color = util.ColorSeg(pred_anchor_mat, gt_mat)
    
    fig, ax = plt.subplots(2,b+1,figsize=((b+1)*2,2*2))
    for i in range(b):
        x_mat = get_sample(x, "latent", idx=i)
        pred_mat = get_sample(pred, "pred", idx=i)
        pred_color = util.ColorSeg(pred_mat, gt_mat)
        
        ax[0,i].imshow(x_mat, cmap="gray"),ax[0,i].axis("off")
        ax[1,i].imshow(pred_color, cmap="gray"),ax[1,i].axis("off")
    
    ax[0,b].imshow(anchor_mat, cmap="gray"),ax[0,b].axis("off")
    ax[1,b].imshow(pred_anchor_color, cmap="gray"),ax[1,b].axis("off")
    
    fig.tight_layout(pad=0.1)
    plt.show()


class Size_Adaptive_Test:
    def __init__(self, im):
        super(Size_Adaptive_Test, self).__init__()
        
        self.softmax = nn.Softmax2d()
        self.factor = 128
        self.h, self.w = im.shape
        self.x = self.fit_tensor(util.ImageRescale(im, [0,1]))
    
    def fit_tensor(self, im):
        dim = len(im.shape)
        size_x = self.h + (self.factor - self.h % self.factor)
        size_y = self.w + (self.factor - self.w % self.factor)
        if dim == 2:
            temp = np.zeros((size_x, size_y), dtype=np.float32)
            temp[:self.h,:self.w] = im
            x_tensor = torch.tensor(temp[None,None,:,:]).type(torch.FloatTensor)
        else:
            raise ValueError
        return x_tensor
    
    def get_prediction(self, y, t_type):
        if t_type == "pred":
            pred_tensor = torch.argmax(self.softmax(y), dim=1)
            matrix = pred_tensor[0,:self.h,:self.w].detach().cpu().numpy()
        elif t_type == "latent":
            matrix = y[0,0,:self.h,:self.w].detach().cpu().numpy()
        else:
            raise ValueError
        return matrix

    def model_test(self, model, device):    
        x = Variable(self.x).to(device)
        y, _ = model(x)
        pred = self.get_prediction(y, "pred")
        return pred


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_inner = checkpoint['lr_sched']

    return model, optimizer, epoch, scheduler_inner
