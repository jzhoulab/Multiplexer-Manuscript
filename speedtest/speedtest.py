import math
import pyfasta
import torch
from torch import nn
import numpy as np
import h5py
import seaborn
import time


genome = pyfasta.Fasta('../resources/hg19.fa')

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

model = Beluga()
model.load_state_dict(torch.load('../resources/deepsea.beluga.pth'))
model.eval()

 
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


class POSMultiplexer(nn.Module):
    def __init__(self):
        super(POSMultiplexer, self).__init__()
        
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
                nn.Conv1d(1280,2002, 1, dilation=1, padding=0),
                nn.BatchNorm1d(2002),
                nn.ReLU(), 
                nn.Conv1d(2002,2002, 1, dilation=1, padding=0))
        
        
            
        

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).to(device)
        encoding = encoding.repeat(x.shape[0], 1, 1)
        x = torch.cat((x.cuda(), encoding.cuda()), dim = 1)
        l_one_output = self.model_one(x)
        l_two_output = self.model_two(l_one_output)
        l_three_output = self.model_three(l_two_output)
        l_four_output = self.model_four(l_three_output)
        l_five_output = self.model_five(l_four_output)
        l_six_output = self.model_six(l_three_output + l_five_output )
        final_out = self.model_final(l_two_output + l_six_output)
     
        return final_out
    

  
#this is pooling version     
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
        x = torch.cat((x.cuda(), encoding.cuda()), dim = 1)
        layer_one = self.model_one(x)
        layer_two = self.model_two(layer_one)
        layer_three = self.model_three(layer_two)
        layer_four = self.model_four(layer_three)
        layer_five = self.model_five(layer_four)
        layer_six= self.model_six(layer_three + layer_five)
        final_out = self.model_final(layer_two + layer_six)        
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out
    
    
    

    
size = []    
for i in CHRS:
    length = len(genome[i])
    size.append(length) 
size_normalized = size/np.sum(size)


def mut_rand_seq():
    """
    returns a random encoded sequence with the  2000bp mutated
    

    """
    chrome_num = np.random.choice(CHRS, p = size_normalized)
    pos = np.random.randint(999, len(genome[chrome_num]) - 1000)    
    seq = genome.sequence({'chr': chrome_num, 'start': pos - 999 , 'stop': pos + 1000})



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
    #copy sequences
 
    seq_encoded_tile = np.tile(seq_encoded, (6000, 1, 1)) #changes to 6000, 4, 2000
    
    #j ranges from pos - 99 to pos + 100
    i = 0
    for j in range(2000):
        #for each element in the original sequence, the next three "layers" of 
        #seq_encoded_tile is a mutation

        seq_encoded_tile[i, :, j] = mydict[mutation_dict[seq[j]][0]]
        seq_encoded_tile[i + 1, :, j] = mydict[mutation_dict[seq[j]][1]]
        seq_encoded_tile[i + 2, :, j] = mydict[mutation_dict[seq[j]][2]]
        i += 3
        
        
    return  seq_encoded_tile
    

    
    
    
    
    
def gen_beluga_input(num_seqs):
    ret_arr = np.zeros((num_seqs, 6000, 4, 2000))
    for i in range(num_seqs):
        ret_arr[i, :, :, :] = mut_rand_seq()
    return ret_arr
        
    
def rand_seq():
    """
       returns randomly encoded sequence (not mutated)
    """
    
    chrome_num = np.random.choice(CHRS, p = size_normalized)
    pos = np.random.randint(999, len(genome[chrome_num]) - 1000)    
    seq = genome.sequence({'chr': chrome_num, 'start': pos - 999 , 'stop': pos + 1000})


    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    

    
    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]
  
        
        
    return seq_encoded


def gen_BM_input(num_seqs):
    ret_arr = np.zeros((num_seqs, 4, 2000))
    for i in range(num_seqs):
        ret_arr[i, :, :] = rand_seq()
        
    
    return torch.from_numpy(ret_arr)









#initialize models
B = Beluga()
B.cuda().eval()

POS_model = POSMultiplexer()
POS_model.cuda().eval()


FULL_model = BelugaMultiplexer()
FULL_model.cuda().eval()



start = torch.cuda.Event(enable_timing=True) 
end = torch.cuda.Event(enable_timing=True)

start2 = torch.cuda.Event(enable_timing=True) 
end2 = torch.cuda.Event(enable_timing=True)

start3 = torch.cuda.Event(enable_timing=True) 
end3 = torch.cuda.Event(enable_timing=True)

btime = 0
POS_model_time = 0
FULL_model_time = 0
runs = 10
sample_size = 64
#run the time test 3-times
for run in range(1,runs + 1):
    #time for Beluga sequence
    input_list = []

    for i in range(sample_size):
        input_list.append(torch.from_numpy(mut_rand_seq()).float().unsqueeze(2).cuda())

    input_list = torch.vstack(input_list)


    beluga_output = []
    start.record()
    with torch.no_grad():
        for i in range(sample_size):
            for j in range(0,6000,sample_size):
                B(input_list[j:j+sample_size])
    end.record()
    torch.cuda.synchronize()
    btime += start.elapsed_time(end)
    

    
    
    
    x = gen_BM_input(sample_size)

    
    
    start2.record()
    with torch.no_grad():
        POS_model(x.float())

        
    end2.record()
    torch.cuda.synchronize()
    POS_model_time += start2.elapsed_time(end2)
    
    
    start3.record()
    with torch.no_grad():
        FULL_model(x.float())
        
    end3.record()
    torch.cuda.synchronize()
    FULL_model_time += start3.elapsed_time(end3)
    


print("Beluga time: ", btime/runs)
print("Position Multiplexer time: ", POS_model_time/runs)
print("Full Multiplexer time: ", FULL_model_time/runs)
print(" ")

