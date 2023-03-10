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

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
def log_fold(alt, ref):
    

    e = 10**(-6)
    top = (alt + e)*(1 - ref + e)
    bot = (1 - alt + e) * (ref + e)
    return np.log(top/bot)



class BelugaMultiplexer(nn.Module):
    def __init__(self):
        super(BelugaMultiplexer, self).__init__()
        
        self.model_one = nn.Sequential(
                nn.Conv1d(5,640, 8, padding = 3),
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).to(device)
        encoding = encoding.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, encoding), dim = 1)
        layer_one = self.model_one(x)
        layer_two = self.model_two(layer_one)
        layer_three = self.model_three(layer_two)
        layer_four = self.model_four(layer_three)
        layer_five = self.model_five(layer_four)
        layer_six= self.model_six(layer_three + layer_five)
        final_out = self.model_final(layer_two + layer_six)        
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out


    
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
    


    seq_encoded = encode_seq(seq)
    seq_encoded_tile = np.tile(seq_encoded, (8000, 1, 1)) #changes to 8000, 4, 2000
    
    for j in range(len(seq)):
        #for each element in the original sequence, the next three "layers" of 
        #seq_encoded_tile is a mutation
        i = j*4
        seq_encoded_tile[i, :, j] = mydict["A"]
        seq_encoded_tile[i + 1, :, j] = mydict["G"]
        seq_encoded_tile[i + 2, :, j] = mydict["C"]
        seq_encoded_tile[i + 3, :, j] = mydict["T"]
        
        
    return torch.from_numpy(seq_encoded_tile).float()
    
    

    




BM = BelugaMultiplexer().cuda()
BM.load_state_dict(torch.load('../../data/comparison/weights/BelugaMultiplexerWeights.pth'))
BM = nn.DataParallel(BM)
BM.eval()


size = 2500
    
inputs = torch.load("../../data/comparison/inputs/correlation_inputs")

bm = torch.zeros((size, 2002, 4, 2000))
#start the loop here
for i in range(size):
    input = torch.from_numpy(inputs[i]).unsqueeze(0).cuda().float()




    ####Best Model section####

    best_output = BM(input).detach().cpu()
    bm[i] = best_output[0]
    
    

torch.save(bm, "../predictions/Multiplexer_predictions")
    



