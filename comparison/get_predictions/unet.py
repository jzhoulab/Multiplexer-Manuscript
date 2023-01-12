import math
import pyfasta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

genome = pyfasta.Fasta('../../data/hg19.fa')

    
def log_fold(alt, ref):
    

    e = 10**(-6)
    top = (alt + e)*(1 - ref + e)
    bot = (1 - alt + e) * (ref + e)
    return np.log(top/bot)




    
def encode_seq(seq):
    """
    returns an encoded sequence 
    
    Args:
        seq: 2000bp sequence
    
    Returns:
        4 x 2000 np.array

    """

    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    
    #this dictionary returns a list of possible mutations for each nucleotide
    mutation_dict = {'a': ['g', 'c', 't'], 'A':['g', 'c', 't'],
                    'c': ['g', 'a', 't'], 'C':['g', 'a', 't'],
                    'g': ['a', 'c', 't'], 'G':['a', 'c', 't'],
                    't': ['g','c', 'a'], 'T':['g', 'c', 'a'],
                    'n': ['n', 'n', 'n'], 'N':['n', 'n', 'n'],
                    '-': ['n', 'n', 'n']}
    
    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]


        
    return torch.from_numpy(seq_encoded)
        

def mutate_seq(seq):
    """
    returns an encoded sequence and every possible mutation
    
    Args:
        seq: 2000bp sequence (encoded)
    
    Returns:
        6000bp mutations encoded to a 6000 x 4 x 2000 np.array

    """

    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    
    #this dictionary returns a list of possible mutations for each nucleotide
    mutation_dict = {'a': ['g', 'c', 't'], 'A':['g', 'c', 't'],
                    'c': ['g', 'a', 't'], 'C':['g', 'a', 't'],
                    'g': ['a', 'c', 't'], 'G':['a', 'c', 't'],
                    't': ['g','c', 'a'], 'T':['g', 'c', 'a'],
                    'n': ['n', 'n', 'n'], 'N':['n', 'n', 'n'],
                    '-': ['n', 'n', 'n']}
    

    seq_encoded = encode_seq(seq)
    seq_encoded_tile = np.tile(seq_encoded, (6000, 1, 1)) #changes to 6000, 4, 2000
    
    for j in range(len(seq)):
        #for each element in the original sequence, the next three "layers" of 
        #seq_encoded_tile is a mutation
        i = j*3
        seq_encoded_tile[i, :, j] = mydict[mutation_dict[seq[j]][0]]
        seq_encoded_tile[i + 1, :, j] = mydict[mutation_dict[seq[j]][1]]
        seq_encoded_tile[i + 2, :, j] = mydict[mutation_dict[seq[j]][2]]
        
        
    return torch.from_numpy(seq_encoded_tile).float()
    
    
   
    
class ConvUnet(nn.Module):
    def __init__(self):
        super(ConvUnet, self).__init__()
        
        self.model_one = nn.Sequential(
                nn.Conv1d(4,640, 8, padding = 3),
                nn.BatchNorm1d(640),
                nn.ReLU(),
                nn.Conv1d(640,640, 8, padding = 4),
                nn.BatchNorm1d(640),
                nn.ReLU())
        
        self.model_two = nn.Sequential(
             
                nn.Conv1d(640,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU())
                
        self.model_three = nn.Sequential(
            
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_four = nn.Sequential(
                #optional (may improve performance)
                nn.Conv1d(1280, 1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_five = nn.Sequential(
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_six = nn.Sequential(
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU())
            
        self.model_final = nn.Sequential(               
                nn.Conv1d(1280,8008, 1, dilation=1, padding=0),
                nn.BatchNorm1d(8008),
                nn.ReLU(), 
                nn.Conv1d(8008,8008, 1, dilation=1, padding=0))
        
        
            
        

    def forward(self, x):
        l_one_output = self.model_one(x)
        l_two_output = self.model_two(l_one_output)
        l_three_output = self.model_three(l_two_output)
        l_four_output = self.model_four(l_three_output)
        l_five_output = self.model_five(l_four_output)
        l_six_output = self.model_six(l_three_output + l_five_output )
        final_out = self.model_final(l_two_output + l_six_output)
        
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out
    
    

####Define Models####

ConvUnet = ConvUnet().cuda()
ConvUnet = nn.DataParallel(ConvUnet)
ConvUnet.load_state_dict(torch.load('../../data/comparison/weights/ConvUnet.pth'))
ConvUnet.eval()


size = 2500

    
inputs = torch.load("../../data/comparison/inputs/correlation_inputs")
#start the loop here
#
CU = torch.zeros((size, 2002, 4, 2000))

for i in range(size):
    input = torch.from_numpy(inputs[i]).unsqueeze(0).cuda().float()



    ####Conv Unet section####
    CU_output = ConvUnet(input).detach().cpu()
    CU[i] = CU_output[0]
    
    
    

torch.save(CU, "../predictions/CU_predictions")
