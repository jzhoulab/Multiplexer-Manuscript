import argparse
import math
import pyfasta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import h5py


np.random.seed(10)
torch.manual_seed(5)


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



 
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


def all_mutations(pos, chrome_num):
    """
    returns an encoded sequence and every possible mutation
    
    Args:
        pos: center of the sequence
    
    Returns:
        6000 mutations encoded to a 8000 x 4 x 2000 np.array

    """
    #'chr1' is the specified chromosome for now
    seq = genome.sequence({'chr': chrome_num, 'start': pos - 999 , 'stop': pos + 1000})



    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    
    #this dictionary returns a list of possible mutations for each nucleotide
    mutation_dict = {'a': ['a','g', 'c', 't'], 'A':['a','g', 'c', 't'],
                    'c': ['a','g', 'c', 't'], 'C':['a','g', 'c', 't'],
                    'g': ['a','g', 'c', 't'], 'G':['a','g', 'c', 't'],
                    't': ['a','g', 'c', 't'], 'T':['a','g', 'c', 't'],
                    'n': ['n', 'n', 'n', 'n'], 'N':['n', 'n', 'n', 'n'],
                    '-': ['n', 'n', 'n', 'n']}
    
    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]
    #copy sequences
    seq_encoded_copy = np.copy(seq_encoded)
    seq_encoded_tile = np.tile(seq_encoded, (8000, 1, 1)) #changes to 8000, 4, 2000
    
    for j in range(len(seq)):
        #for each element in the original sequence, the next three "layers" of 
        #seq_encoded_tile is a mutation
        i = j*4
        seq_encoded_tile[i, :, j] = mydict[mutation_dict[seq[j]][0]]
        seq_encoded_tile[i + 1, :, j] = mydict[mutation_dict[seq[j]][1]]
        seq_encoded_tile[i + 2, :, j] = mydict[mutation_dict[seq[j]][2]]
        seq_encoded_tile[i + 3, :, j] = mydict[mutation_dict[seq[j]][3]]
        
        
    return seq_encoded_copy, seq_encoded_tile
    
    

def log_fold(alt, ref):
    """
    Returns the log fold of a,b
    
    returns log(((alt+1e-6) * (1-ref+1e-6)) /((1-alt+1e-6) * (ref+1e-6)) 
    """
    e = 10**(-6)
    top = (alt + e)*(1 - ref + e)
    bot = (1 - alt + e) * (ref + e)
    return np.log(top/bot)


size = []    
for i in CHRS:
    length = len(genome[i])
    size.append(length) 
size_normalized = size/np.sum(size)


    
def training_data(batch_size, model):
    """
    generates 1 training example. The input is a randomly generated chromosome and the label is 2002
    each sequence has a probability of being selected proportional to it's length
    
    Args:
        batch_size: size of batch that model takes in
    
    
    Returns:
        Output is a random sequence 4 * 2000 np.array of training data and an array of labels. The labels are generated
        by mutation the original 2000bp sequence, encoding it, and predict it's effects by applying the Beluga model
    
    """
  
    model = model
    model.eval()
    

    
    #Choose random chromosome for training data - must not be 'chr8' or 'chr9' and must have >10 "N" values
    seq_encoded_array = np.zeros((1, 4, 2000))
    training_chromosome = np.random.choice(CHRS, p = size_normalized)
    while training_chromosome == "chr8" or training_chromosome == "chr9": #ensures training data is not gathered from chr8 or chr9
        training_chromosome = np.random.choice(CHRS, p = size_normalized)

    N_count = 11
    while N_count > 10:
    #generate a 2000bp sequencr from training_chromosome
        pos = np.random.randint(999, len(genome[training_chromosome]) - 1000)
        seq = genome.sequence({'chr': training_chromosome, 'start': pos - 999 , 'stop': pos + 1000})
    #checks number of N's in the sequence is less than 10
        N_count = seq.count("N") 
            
    #this encodes the sequence, as well as generates a mutated version of the sequence
    #orig_arr_encoded is completed, mut_arr_encoded needs to be run through beluga
    train_input_encoded, mut_arr_encoded = all_mutations(pos, training_chromosome)
    
    
    model.cuda()
    input = torch.from_numpy(train_input_encoded).unsqueeze(1)
    input = input.unsqueeze(0).float().cuda()
    dummy = model.forward(input).cpu().detach().numpy().copy()
    
 
    mutations_predict_arr = []
    for i in range(int(math.floor(8000/batch_size)) + 1):
        inputs = mut_arr_encoded[i*batch_size : (i+1)*batch_size, :, :] 
        input = torch.from_numpy(inputs).unsqueeze(2)
        input = input.cuda().float()
        mutations_predict_arr.append(model.forward(input).cpu().detach().numpy().copy())   
    mutations_predict_arr = np.vstack(mutations_predict_arr)
    
    

    
    
    reference = dummy
    alternative = mutations_predict_arr
        
    
    return train_input_encoded, log_fold(alternative, reference)



def validation_data(batch_size, model):
    """
    Generates one sample of validation data with input and labels. 
    
    Returns:
        One validation sample. The Label is generated by generating a 6000*4*2000 mutated array and running the
        mutated array through the Beluga model
        
    
    """
    model = model
    model.eval()
 
 
    
    #generate validation data - must be from "chr8" or "chr9" and have 'N' count less than 10
    val_probability = [len(genome["chr8"]), len(genome["chr9"])]
    val_probability = val_probability/np.sum(val_probability)
    val_chromosome = np.random.choice(["chr8", "chr9"], p = val_probability)
    
    val_n_count = 11
    while val_n_count > 10:
        val_pos = np.random.randint(999, len(genome[val_chromosome]) - 1000)
        val_seq = genome.sequence({'chr': val_chromosome, 'start': val_pos - 999 , 'stop': val_pos + 1000})
        val_n_count = val_seq.count("N") 
            
    val_data, val_mut_encoded = all_mutations(val_pos, val_chromosome)
    #val_data is ready to be returned
    
    #Call Beluga on Val_data
    #model = nn.DataParallel(model)
    model.cuda()
    input = torch.from_numpy(val_data).unsqueeze(1)
    input = input.unsqueeze(0).cuda().float()
    dummy = model.forward(input).cpu().detach().numpy().copy()
    
    
    
    
    #Call Beluga on Val_mut_encoded and return the result
    val_labels = []
    for i in range(int(math.floor(8000/batch_size)) + 1):
        model.cuda()
        input = val_mut_encoded[i*batch_size : (i+1)*batch_size, :, :]
        input = torch.from_numpy(input).unsqueeze(2)
        input = input.cuda().float()
        val_labels.append(model.forward(input).cpu().detach().numpy().copy())   
    val_labels = np.vstack(val_labels)

  

    return val_data, log_fold( val_labels, dummy)
    

    

def gen_training_data(num_seqs, model):
    """
    Generates num_seqs # of training samples by calling gen_training_data
    
    Args:
        num_seqs: the number of training samples generated by the method
        
    Returns:
        model_input_arr: an array of model inputs
        label_arr: an array of labels for the model input data
    
    """
    model_input_arr = np.zeros((num_seqs, 4, 2000))
    label_arr = np.zeros((num_seqs, 8000, 2002))
    for i in range(num_seqs):
        input_sample, label = training_data(192, model)
        model_input_arr[i, :, :] = input_sample
        label_arr[i, :, :] = label
        
        
    return torch.from_numpy(model_input_arr).float().cuda(), torch.from_numpy(label_arr).float().cuda()
    
    
    
def gen_validation_data(num_seqs, model):
    """
    Generates num_seqs # of training samples by calling gen_validation_data
    
    Args:
        num_seqs: the number of sequence the method generates
    
    Returns:
        val_input_arr: an array of num_seqs validation inputs
        val_label_arr an array of num_seqs validation labels
    """
    val_input_arr = np.zeros((num_seqs, 4, 2000))
    val_label_arr = np.zeros((num_seqs, 8000, 2002))
    
    for i in range(num_seqs):
        input_sample, label = validation_data(192, model)
        val_input_arr[i, :, :] = input_sample
        val_label_arr[i, :, :] = label
        
    return torch.from_numpy(val_input_arr).float().cuda(), torch.from_numpy(val_label_arr).float().cuda()
  




class Interpreter_Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = x.shape[0]
        
    def __getitem__(self, index):
        sample = self.x[index,:,:], self.y[index, :, :]
        return sample
    
    def __len__(self):
        return self.length
    
    
Beluga_model = Beluga()
Beluga_model.load_state_dict(torch.load('../../data/deepsea.beluga.pth'))
Beluga_model.cuda()
Beluga_model = nn.DataParallel(Beluga_model)

    
start = time.time()
val_arr, val_labels_arr = gen_validation_data(288, Beluga_model )
end = time.time()
validation_data_obj = Interpreter_Data(val_arr, val_labels_arr)
val_loader = DataLoader(dataset = validation_data_obj, batch_size = 16)
print("time to generate 288 samples ", end - start) 


class BelugaInterpreter(nn.Module):
    def __init__(self):
        super(BelugaInterpreter, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(4,640, 8, padding = 3),
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

    


def train( val_data, model, gen_model, optimizer, loss_function, epochs):

    min_validation_loss = 9999
    print(next(model.parameters()).is_cuda)
    for epoch in range(epochs):
        for i in range(200):
            model.train()
            x,y = gen_training_data(16, gen_model ) 
            loss_sum = 0
            optimizer.zero_grad()
            y = y.transpose(1,2)
            y = torch.reshape(y, (y.shape[0], 2002, 2000, 4))
            y = y.transpose(2,3)
            
            #concat position encoding

            yhat = model.forward(x)

            loss = loss_function(yhat, y)
            loss_sum += loss.detach()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 and i == 198:
                print("Training loss on Epoch ", epoch, "is ", loss_sum.item())

            
        #validation check    
        #run prediction batch by batch - 32 at a time
        if epoch % 5 == 0:
            loss_array = []
            model.eval()
            loss = 0
            for x,y in val_data:
                with torch.no_grad():

                    yhat = model.forward(x)
    
                    y = y.transpose(1,2)
                    y = torch.reshape(y, (y.shape[0], 2002, 2000, 4))
                    y = y.transpose(2,3)
                    loss += loss_function(yhat, y) 
                    
                    if loss.item() < min_validation_loss:
                        min_validation_loss = loss.item()
                        torch.save(model.state_dict(), "Conv.pth")
                        
                        print("saved state dict!")
                        
            
                    
            print("Validation loss on epoch ", epoch, "is ", loss.item())
            
            
            #torch.save(optimizer.state_dict(), "final.pth")
           
                

    return loss_array


BI = BelugaInterpreter() 
BI.cuda()
BI = nn.DataParallel(BI)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(BI.parameters(), lr = 0.0001)    
f = train( val_loader, BI, Beluga_model, optimizer, loss_function, 99999999) 