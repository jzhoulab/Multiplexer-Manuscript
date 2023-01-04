import torch
import numpy as np
import math

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))






beluga = torch.load("../zerod_data/zerod_beluga") #torch.Size([2500, 2000, 3, 2002])
multiplexer = torch.load("../zerod_data/zerod_multiplexer") #torch.Size([2500, 2000, 3, 2002])
             

correlation_list = []

for i in range(2002):
    A_profile = beluga[ :, :, :, i].mean(axis = 2)
    B_profile = multiplexer[ :, :,:,  i].mean(axis = 2)
    
    corre = corr(A_profile.numpy().flatten(), B_profile.numpy().flatten(), np.abs(A_profile.numpy().flatten()))
    correlation_list.append(corre)
    
    
print(sum(correlation_list)/len(correlation_list))
    