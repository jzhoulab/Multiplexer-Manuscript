import pyfasta
import torch
import numpy as np



genome = pyfasta.Fasta('../resources/hg19.fa')

 
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


def encode(pos, chrome_num):
    """
    returns an encoded sequence from pos, chrome_num

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
    

    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]
    
        
    return seq_encoded, seq
    
    
    
def gen_inputs(num_inputs):
    inputs_list = []
    seq_strings = []
    for i in range(num_inputs):    
    
        size = []    
        for i in CHRS:
            length = len(genome[i])
            size.append(length) 
        size_normalized = size/np.sum(size)


        #generate validation data - must be from "chr8" or "chr9" and have 'N' count less than 10
        val_probability = [len(genome["chr8"]), len(genome["chr9"])]
        val_probability = val_probability/np.sum(val_probability)
        val_chromosome = np.random.choice(["chr8", "chr9"], p = val_probability)

        val_n_count = 11
        while val_n_count > 10:
            val_pos = np.random.randint(999, len(genome[val_chromosome]) - 1000)
            val_seq = genome.sequence({'chr': val_chromosome, 'start': val_pos - 999 , 'stop': val_pos + 1000})
            val_n_count = val_seq.count("N") 

        val_data, seq = encode(val_pos, val_chromosome)
        inputs_list.append(val_data)
        seq_strings.append(seq)
        
    return inputs_list, seq_strings
    
    
x,y = gen_inputs(2500)

torch.save(x, "correlation_inputs")
torch.save(y, "inputs_as_strings")