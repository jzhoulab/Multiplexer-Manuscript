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

def encoding_to_sequence(x):
    ret_string = ""
    for i in range(2000):
        if x[ 0, i] == 1:
            ret_string += "a"
            
        elif x[ 1, i] == 1:
            ret_string += "g"
            
            
        elif x[ 2, i] == 1:
            ret_string += "c"
            
            
        elif x[ 3, i] == 1:
            ret_string += "t"
        
        else:
            print("none change")

            
    return ret_string




Beluga = Beluga().cuda()
Beluga.load_state_dict(torch.load('../../data/deepsea.beluga.pth'))
Beluga.eval()

BM = BelugaMultiplexer().cuda()
BM.load_state_dict(torch.load('../../data/comparison/weights/BelugaMultiplexerWeights.pth'))
BM.eval()



def get_index(char):
    "returns index of char and index of comp"
    
    c = char.lower()
    if c == "a":
        return 0,3
    if c == "g":
        return 1,2
    if c == "c":
        return 2,1
    if c == "t":
        return 3,0
    
####MAKE PREDICTIONS WITH BELUGA####

ref_inputs = torch.load("./AtacSeqData/encoded_ref_seqs")
alt_inputs = torch.load("./AtacSeqData/encoded_alt_seqs")
rc_ref_inputs = torch.load("./AtacSeqData/encoded_refs_reverse_comps")
rc_alt_inputs = torch.load("./AtacSeqData/encoded_alts_reverse_comps")

beluga_ref_values = []
beluga_alt_values = []

for i in range(len(ref_inputs)):
    ref = ref_inputs[i]
    alt = alt_inputs[i]
    rc_ref = rc_ref_inputs[i]
    rc_alt = rc_alt_inputs[i]

    

    
    
    with torch.no_grad():
        ref_beluga_output = Beluga(ref.unsqueeze(0).unsqueeze(2).cuda().float())
        alt_beluga_output = Beluga(alt.unsqueeze(0).unsqueeze(2).cuda().float())
        rc_ref_beluga_output = Beluga(rc_ref.unsqueeze(0).unsqueeze(2).cuda().float())
        rc_alt_beluga_output = Beluga(rc_alt.unsqueeze(0).unsqueeze(2).cuda().float())        
        
    #53 is cell type CMK
    ref_avg = (ref_beluga_output[0, 53].item() + rc_ref_beluga_output[0, 53 ].item())/2
    alt_avg = (alt_beluga_output[0, 53].item() + rc_alt_beluga_output[0, 53 ].item())/2
    
    beluga_ref_values.append(ref_avg)
    beluga_alt_values.append(alt_avg)
    

        
torch.save(beluga_ref_values, "./AtacSeqData/beluga_out_refs")
torch.save(beluga_alt_values, "./AtacSeqData/beluga_out_alts")


data = pd.read_csv('../../data/experimental/AtacSeq.csv')
ref_seqs = []
alt_seqs = []
alt_is_open_positions = [] #a list of indices where the alt is the open allele

c = 2
for i in range(45071):
    CHR = "chr" + str(data["chr"][i])
    POS = data["position"][i]
    seq = genome.sequence({'chr': CHR, 'start': POS - 999 , 'stop': POS + 1000})
    
    
    try:
        if seq[999].lower() == data["consensus open allele"][i].lower():
            alt_seq = seq[:999] + data["consensus closed allele"][i] + seq[1000:]

        elif seq[999].lower() == data["consensus closed allele"][i].lower():
            alt_seq = seq[:999] + data["consensus open allele"][i] + seq[1000:]
            alt_is_open_positions.append(i)

        else:
            print("error")
    except:
        print(i+c)
        c = c-1


    alt_seqs.append(alt_seq)
    ref_seqs.append(seq)
    
torch.save(ref_seqs, "./AtacSeqData/ref_seqs")    
torch.save(alt_seqs, "./AtacSeqData/alt_seqs")   

ref_prediction = torch.load("./AtacSeqData/beluga_out_refs")
alt_prediction = torch.load("./AtacSeqData/beluga_out_alts")
            
####DETERMINE ACCURACY OF PREDICTIONS####
total_samples = []
proportion_list = []
margin_list = np.linspace(0, 0.1, 1000)


for margin in margin_list:
    total = 0
    count = 0

    for i in range(45071):
        alt = alt_prediction[i]
        ref = ref_prediction[i]

        if np.abs(ref-alt) > margin:
            total += 1
            
            
            if i in alt_is_open_positions and alt_prediction[i] > ref_prediction[i]:
                count += 1
                
            elif (not i in alt_is_open_positions) and alt_prediction[i] < ref_prediction[i]:
                count += 1
                
    if total > 0:            
        proportion_list.append(count/total)
        total_samples.append(total)

Beluga_accuracy = {"accuracy": proportion_list, "margins": margin_list, "num_samples": total_samples}
torch.save(Beluga_accuracy, "./AtacSeqData/Beluga_accuracy")


#### GET Multiplexer PREDICTIONS ####
ref_inputs = torch.load("./AtacSeqData/encoded_ref_seqs")
alt_inputs = torch.load("./AtacSeqData/encoded_alt_seqs")
ref_prediction = torch.load("./AtacSeqData/beluga_out_refs")
ref_rc_inputs = torch.load("./AtacSeqData/encoded_refs_reverse_comps")
alt_rc_inputs = torch.load("./AtacSeqData/encoded_alts_reverse_comps")


Full_alt_predictions = []
for i in range(45070):
    ref_input = ref_inputs[i]
    rc_input = ref_rc_inputs[i]
    
    
    #get ref prediction to be used in log_unfold
    with torch.no_grad():
        ref_beluga_output = Beluga(ref_input.unsqueeze(0).unsqueeze(2).cuda().float()).cpu()
        rc_beluga_output = Beluga(rc_input.unsqueeze(0).unsqueeze(2).cuda().float()).cpu()
    
    encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).cuda()
    encoding = encoding.repeat(1, 1, 1)
    
    ref_input = torch.cat((ref_input.unsqueeze(0).cuda().float(), encoding), dim = 1)
    rc_input = torch.cat((rc_input.unsqueeze(0).cuda().float(), encoding), dim = 1)
    
    with torch.no_grad():
        ref_out = BM(ref_input).cpu()
        rc_out = BM(rc_input).cpu()
        
    if i in alt_is_open_positions:
        index, comp = get_index(data["consensus open allele"][i])
    else:
        index, comp = get_index(data["consensus closed allele"][i])
        

   
    alt_from_refs = log_unfold(ref_beluga_output[0,:], ref_out[0, :, index, 999 ])   
    alt_from_rc = log_unfold(rc_beluga_output[0, :], rc_out[0, :, comp, 999 ])

    #53 is 
    alt_pred = (alt_from_refs[53] + alt_from_rc[53])/2
    
    Full_alt_predictions.append(alt_pred.detach().cpu())
 
    
torch.save(Full_alt_predictions, "./AtacSeqData/BM_alt_predictions")


ref_prediction = torch.load("./AtacSeqData/beluga_out_refs")
alt_prediction = torch.load("./AtacSeqData/BM_alt_predictions")


####DETERMINE ACCURACY OF PREDICTIONS####
total_samples = []
proportion_list = []
margin_list = np.linspace(0, 0.10, 1000)


for margin in margin_list:
    total = 0
    count = 0


    for i in range(45070):
        
        alt = alt_prediction[i] 
        ref = ref_prediction[i]
        diff = alt - ref
        if np.abs(diff) > margin :
            total += 1
            
        
            if i in alt_is_open_positions and alt_prediction[i] > ref_prediction[i]:
                count += 1
                
            elif (not i in alt_is_open_positions) and alt_prediction[i] < ref_prediction[i]:
                count += 1
      


                
    if total > 10:            
        proportion_list.append(count/total)
        total_samples.append(total)


Full_accuracy = {"accuracy": proportion_list, "margins": margin_list[:len(total_samples)], "num_samples": total_samples}

torch.save(Full_accuracy, "./AtacSeqData/BM_accuracy")