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





A = torch.load("../predictions/Beluga_predictions") #shape 2500x8000x2002

B = torch.load("../predictions/Multiplexer_predictions").transpose(2,3).reshape(-1, 2002, 8000).transpose(1,2) #shape 
#B = torch.load("../predictions/Multiplexer_predictions_small").transpose(2,3).reshape(-1, 2002, 8000).transpose(1,2) #shape 2500x2002x4x2000 -> 2500x8000x2002


correlation_list = []

for i in range(2002):
    A_profile = A[ :, :, i]
    B_profile = B[ :, :, i]
    
    corre = corr(A_profile.numpy().flatten(), B_profile.numpy().flatten(), np.abs(A_profile.numpy().flatten()))
    correlation_list.append(corre)
    
    
torch.save(correlation_list, "full_correlation")

print(sum(correlation_list)/len(correlation_list))


