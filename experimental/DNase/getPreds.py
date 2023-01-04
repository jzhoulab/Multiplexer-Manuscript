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
    

    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]


        
    return torch.from_numpy(seq_encoded)


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
        l_one_output = self.model_one(x)
        l_two_output = self.model_two(l_one_output)
        l_three_output = self.model_three(l_two_output)
        l_four_output = self.model_four(l_three_output)
        l_five_output = self.model_five(l_four_output)
        l_six_output = self.model_six(l_three_output + l_five_output )
        final_out = self.model_final(l_two_output + l_six_output)
        
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out

    
def log_unfold(ref, lg_diff):
    """
    given the reference value and lg_difference, returns the alternative value
    
    """
    e = 10**(-6)
    
    y = (ref + e)*np.exp(lg_diff)/(1 - ref + e)
    
    return (y)*(1+e)/(1+y)


#define models
Beluga = Beluga().cuda()
Beluga.load_state_dict(torch.load('../../data/deepsea.beluga.pth'))
Beluga.eval()

BM = BelugaMultiplexer().cuda()
BM.load_state_dict(torch.load('../../data/comparison/weights/BelugaMultiplexerWeights.pth'))
BM.eval()






data = pd.read_csv("../../data/experimental/DNasedata.csv")

files_celltype_dict = torch.load("../../data/experimental/files_celltype_dict")
celltype_position_dict = torch.load("../../data/experimental/celltype_position_dict")



ref_inputs = torch.load("./DNaseData/encoded_refs_reverse_comps")
alt_inputs = torch.load("./DNaseData/encoded_alts_reverse_comps")


beluga_ref_values = []
beluga_alt_values = []

####MAKE PREDICTIONS WITH BELUGA####

for i in range(len(ref_inputs)):
    ref = ref_inputs[i]
    alt = alt_inputs[i]

    file_name = data.iloc[i]["File name"]
    cell_type = files_celltype_dict[file_name]
    position = int(celltype_position_dict[cell_type])
    
    with torch.no_grad():
        ref_beluga_output = Beluga(ref.unsqueeze(0).unsqueeze(2).cuda().float())
        alt_beluga_output = Beluga(alt.unsqueeze(0).unsqueeze(2).cuda().float())

    
    beluga_ref_values.append(ref_beluga_output[0, position])
    beluga_alt_values.append(alt_beluga_output[0, position])
    

        
torch.save(beluga_ref_values, "./DNaseData/revcomp_refs")
torch.save(beluga_alt_values, "./DNaseData/revcomp_alts")


ref_inputs = torch.load("./DNaseData/encoded_ref_seqs")
alt_inputs = torch.load("./DNaseData/encoded_alt_seqs")  


beluga_ref_values = []
beluga_alt_values = []

for i in range(len(ref_inputs)):
    ref = ref_inputs[i]
    alt = alt_inputs[i]

    file_name = df.iloc[i]["File name"]
    cell_type = files_celltype_dict[file_name]
    position = int(celltype_position_dict[cell_type])
    
    with torch.no_grad():
        ref_beluga_output = Beluga(ref.unsqueeze(0).unsqueeze(2).cuda().float())
        alt_beluga_output = Beluga(alt.unsqueeze(0).unsqueeze(2).cuda().float())

    
    beluga_ref_values.append(ref_beluga_output[0, position])
    beluga_alt_values.append(alt_beluga_output[0, position])
    

        
torch.save(beluga_ref_values, "./DNaseData/beluga_ref_values")
torch.save(beluga_alt_values, "./DNaseData/beluga_alt_values")


ref_prediction = torch.load("./DNaseData/beluga_ref_values")
alt_prediction = torch.load("./DNaseData/beluga_alt_values")

combo = torch.load("./DNaseData/revcomp_refs")
combo2 = torch.load("./DNaseData/revcomp_alts")

ref_prediction = [(i+j)/2 for (i,j) in zip(ref_prediction, combo)]
alt_prediction = [(i+j)/2 for (i,j) in zip(alt_prediction, combo2)]
            
####DETERMINE ACCURACY OF PREDICTIONS####
total_samples = []
proportion_list = []
margin_list = np.linspace(0, 0.8, 100)
margin_list = margin_list
for margin in margin_list:
    total = 0
    count = 0
    for i in range(len(data["REF"])):
        if np.abs(ref_prediction[i].cpu() - alt_prediction[i].cpu()) > margin:
            total += 1
            if data["Ref reads"][i] > data["Alt reads"][i] and ref_prediction[i] > alt_prediction[i]:
                count += 1
                
            elif data["Ref reads"][i] < data["Alt reads"][i] and ref_prediction[i] < alt_prediction[i]:
                count += 1
                
    if total > 0:            
        proportion_list.append(count/total)
        total_samples.append(total)


Beluga_accuracy = {"accuracy": proportion_list, "margins": margin_list, "num_samples": total_samples}
torch.save(Beluga_accuracy, "./DNaseData/Beluga_accuracy_dict")




ref_inputs = torch.load("./DNaseData/encoded_ref_seqs")
alt_inputs = torch.load("./DNaseData/encoded_alt_seqs")


def get_index(char):
    c = char.lower()
    if c == "a":
        return 0
    if c == "g":
        return 1
    if c == "c":
        return 2
    if c == "t":
        return 3
        
#### GET Multiplexer PREDICTIONS ####

BM_ref_values = []
BM_alt_values = []

for i in range(len(ref_inputs)):
    ref = ref_inputs[i]
    alt = alt_inputs[i]
        
    file_name = data.iloc[i]["File name"]
    cell_type = files_celltype_dict[file_name]
    position = int(celltype_position_dict[cell_type])
    
    with torch.no_grad():
        ref_prediction = Beluga(ref.unsqueeze(0).unsqueeze(2).cuda().float())
        
        encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).cuda()
        encoding = encoding.repeat(1, 1, 1)
        ref = torch.cat((ref.unsqueeze(0).cuda().float(), encoding), dim = 1)
        BM_output = BM(ref).detach().cpu()

        alt_index = get_index(df["ALT"][i])
        
    
        alt_value = log_unfold(ref_prediction[0, :].detach().cpu(), BM_output[0,:, alt_index, 999].detach().cpu())[position].item()



    
    BM_alt_values.append(alt_value)



torch.save(BM_alt_values, "./DNaseData/BM_alt_values")


    
ref_prediction = torch.load("./DNaseData/beluga_ref_values")
alt_prediction = torch.load("./DNaseData/BM_alt_values")

data = pd.read_csv("../../data/experimental/DNasedata.csv")

            
#### GET MULTIPLEXER ACCURACY ####
total_samples = []
proportion_list = []
margin_list = np.linspace(0, 0.8, 50)
for margin in margin_list:
    total = 0
    count = 0
    for i in range(len(data["REF"])):
        if np.abs(ref_prediction[i].cpu() - alt_prediction[i]) > margin:
            total += 1
            if data["Ref reads"][i] > data["Alt reads"][i] and ref_prediction[i] > alt_prediction[i]:
                count += 1
            elif data["Ref reads"][i] < data["Alt reads"][i] and ref_prediction[i] < alt_prediction[i]:
                count += 1
    if total > 0:            
        proportion_list.append(count/total)
        total_samples.append(total)


BM_accuracy = {"accuracy": proportion_list, "margins": margin_list, "num_samples": total_samples}
torch.save(BM_accuracy, "./DNaseData/BM_accuracy_dict")
  
