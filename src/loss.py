# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:10:38 2022

@author: hudew
"""

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class ContrastiveLossELI5(nn.Module):
    
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super(ContrastiveLossELI5, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(self.device))
        self.verbose = verbose
            
    
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
#            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(self.device).scatter_(0, torch.tensor([i]).to(self.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss
    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    

class SSIMLoss(nn.Module):
    def __init__(self,):
        super(SSIMLoss, self).__init__()
    
    def forward(self, inputs, targets):
        N,C,H,W = inputs.size()
        
        inputs_norm = torch.div(inputs, torch.max(inputs))
        targets_norm = torch.div(targets, torch.max(targets))
        
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=C)
        ms_ssim_loss = 1 - ms_ssim_module(inputs_norm, targets_norm)
        
        return ms_ssim_loss
    

def instance_correlation_loss(a, b, lambda_=1, device="cuda:0"):
    N = a.size()[0]
    
    z_a = a.view(N, -1)
    z_b = b.view(1, -1).repeat(N,1)
    
    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
    
    c = torch.matmul(z_a_norm.T, z_b) / N
    D,_ = c.size()
    
    c_diag = torch.pow((c - 1), 2) * torch.eye(D).to(device)
    c_off_diag = c * ((1 - torch.eye(D)) * lambda_).to(device)
    
    loss = (c_diag + c_off_diag).sum()
    
    return loss