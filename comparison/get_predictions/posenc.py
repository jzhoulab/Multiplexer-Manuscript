import argparse
import math
import pyfasta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import h5py
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
    
    
class ConvEnc(nn.Module):
    def __init__(self):
        super(ConvEnc, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(5,640, 8, padding = 3),
                nn.BatchNorm1d(640),
                nn.ReLU(),
                nn.Conv1d(640,640, 8, padding = 4),
                nn.BatchNorm1d(640),
                nn.ReLU(),
             
                nn.Conv1d(640,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
            
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                #optional (may improve performance)
                nn.Conv1d(1280,1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
            
                #finally
                nn.Conv1d(1280,8008, 1, dilation=1, padding=0),
                nn.BatchNorm1d(8008),
                nn.ReLU(), 
                nn.Conv1d(8008,8008, 1, dilation=1, padding=0)
            )
        )

    def forward(self, x):

        final_out = self.model(x)
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out
    
####Define Models####


ConvEncode = ConvEnc().cuda()
ConvEncode = nn.DataParallel(ConvEncode)
ConvEncode.load_state_dict(torch.load('../../data/comparison/weights/ConvEnc.pth'))
ConvEncode.eval()


size = 2500

    
inputs = torch.load("../../data/comparison/inputs/correlation_inputs")
p = torch.zeros((size , 2002, 4, 2000))


for i in range(size ):
    input = torch.from_numpy(inputs[i]).unsqueeze(0).cuda().float()


    ####Conv Encode section####
     
    #set up input
    encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).cuda()
    encoding = encoding.repeat(input.shape[0], 1, 1)
    x = torch.cat((input.cuda(), encoding), dim = 1)


    CE_output = ConvEncode(x).detach().cpu()
    p[i] = CE_output[0]
    



torch.save(p, "../predictions/convEncode_predictions")
#torch.save(a, "BMavg_predictions")
