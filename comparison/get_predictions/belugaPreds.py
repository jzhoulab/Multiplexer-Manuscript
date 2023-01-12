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
    
    

    

####Define Models####
Beluga_model = Beluga()
Beluga_model.load_state_dict(torch.load('../../data/deepsea.beluga.pth'))
Beluga_model = nn.DataParallel(Beluga_model)
Beluga_model.eval().cuda()



size = 2500


    
inputs = torch.load("../../data/comparison/inputs/correlation_inputs")
b = torch.zeros((size, 8000, 2002))
#start the loop here
for i in range(size):
    input = torch.from_numpy(inputs[i]).unsqueeze(0).cuda().float()



    #Format input
    seqs = torch.load("../../data/comparison/inputs/inputs_as_strings")
    BI_input = mutate_seq(seqs[i]).unsqueeze(2).cuda() #ready for Beluga



    #calculate reference output
    reference = Beluga_model(input.unsqueeze(2))



    #calculate alternative output (in batches)
    #define batch size
    batch_size = 800
    alt_arr = []
    for j in range(int(math.floor(8000/batch_size))):
        batched_input = BI_input[j*batch_size : (j+1)*batch_size, :,  :, :] 
        alt_arr.append(Beluga_model(batched_input).cpu().detach())   
    alt_arr = torch.vstack(alt_arr)



    final_beluga_output = log_fold(alt_arr.cpu().detach(), reference.cpu().detach())

    
    b[i] = final_beluga_output
   

    ####Best Model section####

 
    
torch.save(b, "../predictions/Beluga_predictions")

    



