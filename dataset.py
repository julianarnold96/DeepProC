import numpy as np
import pandas as pd
import random
import torch
import dataclasses
from dataclasses import dataclass
from typing import Union 

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


class AminoEncoder(ABC):
    # please follow this order for the embeddings
    AMINOS = 'ARNDCQEGHILKMFPSTWYVBZX'
    AMINO_IDX = {a: i for i, a in enumerate('ARNDCQEGHILKMFPSTWYVBZX')}

    @abstractmethod
    def get_weights_as_torch_tensor(self):
        pass

    @classmethod
    def encode_sequence(cls, sequence):
        return [cls.AMINO_IDX[a] for a in sequence]


class RandomEncoder(AminoEncoder):
    def get_weights_as_torch_tensor(self, size, learnable=True):
        return torch.normal(0, 1, size=(len(self.AMINOS), size), requires_grad=learnable)


class OneHotEncoder(AminoEncoder):
    def get_weights_as_torch_tensor(self):
        return torch.eye(len(self.AMINOS), dtype=torch.float32)


class Blosum50Encoder(AminoEncoder):
    def get_weights_as_torch_tensor(self, learnable=False):
        return torch.tensor([
            [5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, -2, -1],
            [-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3, -1, 0],
            [-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3, 4, 0],
            [-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4, 5, 1],
            [-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -
                3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3],
            [-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3, 0, 4],
            [-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3, 1, 5],
            [0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4, -1, -2],
            [-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4, 0, 0],
            [-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4, -4, -3],
            [-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1, -4, -3],
            [-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3, 0, 1],
            [-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1, -3, -1],
            [-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1, -4, -4],
            [-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -
                1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1],
            [1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2, 0, 0],
            [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0, 0, -1],
            [-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3, -5, -2],
            [-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1, -3, -2],
            [0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5, -4, -3],
            [-2, -1, 4, 5, -3, 0, 1, -1, 0, -4, -4, 0, -3, -4, -2, 0, 0, -5, -3, -4, 5, 2],
            [-1, 0, 0, 1, -3, 4, 5, -2, 0, -3, -3, 1, -1, -4, -1, 0, -1, -2, -2, -3, 2, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=torch.float32, requires_grad=learnable)


@dataclass
class BatchData:
    mask: torch.Tensor
    proteins: torch.Tensor
    labels: torch.Tensor


class Dataset:
    def __init__(self, data, batch_size, shuffle=True, device=None, aa_left = 15, aa_right = 15):
        self.proteins = [
            AminoEncoder.encode_sequence(prot)
            for prot in data['proteins']
        ]

        self.labels = data['label'].values.astype(np.float32)
        self.counts = data['count'].values.astype(np.int)
        self.shuffle = shuffle
        self.device = device
        self.batch_size = batch_size
        self.length = aa_left + aa_right

    def __len__(self):
        return len(self.proteins) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.proteins))
        
        if self.shuffle:
            np.random.shuffle(indices)

        for ibatch in range(len(self)):
            batch_indices = indices[
                ibatch * self.batch_size:(ibatch + 1) * self.batch_size
            ]
            # bin labels hehe
            mask_length, proteins, labels = [], [], []
            
            for i, isample in enumerate(batch_indices):

                #how many members in bag
                if self.counts[isample] == 0:
                    continue
                mask_length.append(self.counts[isample])

                # generate the bags via taking each window

                proteins.extend([

                    self.proteins[isample][self.length*j:self.length*(j+1)]
                    for j in range(self.counts[isample])
                ])

                labels.append(self.labels[isample])

            assert len(proteins) > 0, 'empty batch no proteins'
            assert len(labels) > 0, 'empty batch no labels'

            # build mask
            assert sum(mask_length) == len(proteins)
            batch_mask = np.zeros((len(labels), sum(mask_length)), dtype=np.float32)
            cur = 0
            for i, l in enumerate(mask_length):
                batch_mask[i, cur:cur+l] = 1
                cur += l

            yield BatchData(
                mask=torch.tensor(batch_mask).to(self.device),
                proteins=torch.tensor(proteins).to(self.device),
                labels=torch.tensor(labels).to(self.device),
            )


class KFoldCV:
    '''
    k-fold cross-validation stratified by allele and removing
    duplicates and overlaps (only) from the validation set
    '''

    def __init__(self, k, seed=None):
        self.k = k
        self.seed = seed

    def split(self, data):
        ''' yields pairs of boolean masks for training and validation sets '''
        alleles = data['proteins'].unique().tolist()
        random.seed(self.seed)
        random.shuffle(alleles)

        for i in range(self.k):
            fold_alleles = set(alleles[i::self.k])
            fold_mask = data['proteins'].isin(fold_alleles)

            yield ~fold_mask, fold_mask


def generate_window(protein, start_pos, end_pos, seq_length_left , seq_length_right , aa_left_N=2, aa_right_C=2):

            a = 'Z' * (seq_length_left+aa_left_N) + protein + 'Z' * (seq_length_right+aa_right_C)
            b = start_pos + seq_length_left + aa_left_N - 1
            c = b + (end_pos - start_pos + 1)
                  
            start = b - (seq_length_left + aa_left_N)
            end = c + (seq_length_right + aa_right_C)
                  
            slicing = slice(start, end)
            window = a[slicing]
            #start_pos = seq_length_left + aa_left_N + 1
            #end_pos = seq_length_left + aa_left_N + 1 + (end_pos - start_pos)

            return window


# BAGS

def generate_bags(negatives_generator, data, nterm, seq_length_left , seq_length_right , aa_left_N = 2, aa_right_C = 2):
    
    # data has to be sorted by epitope sequence in order to prevent dublicates
    # data[2]: protein, data[3]: epitope, data[6]: start_pos, data[7]: end_pos

    bags = []
    current_epitope = ''
    count = 0
    #np.sort(data, axis=)
    
    for sample in data:

        if sample[3] == current_epitope:
          count = count + 1
          pass
        if sample[3] != current_epitope:
          current_epitope = sample[3]  
          bag = sampler(sample, current_epitope, data, count, negatives_generator,nterm,seq_length_left, seq_length_right)
          count = count + 1
          if len(bag) > 0:
            bags.append(bag) 

    return bags
    
def sampler(sample, current_epitope, data, count, negatives_generator, nterm,seq_length_left , seq_length_right ):
        
          bag = ''
          proteins = []
          for i in data[count:len(data)]:
            if i[3] != current_epitope:
              return bag

            if i[3] == current_epitope:
              if negatives_generator == 'shuffled_negatives':   
                  window = generate_window(protein=i[2], start_pos=i[6], end_pos=i[7], seq_length_left= seq_length_left, seq_length_right = seq_length_right)
                  proteins.append(window)             
              if negatives_generator == 'shifting_window_negatives':
                  window = generate_window(protein=i[2], start_pos=i[6], end_pos=i[7],seq_length_left= seq_length_left, seq_length_right = seq_length_right)
                  proteins.append(window)                         

              unique_proteins = np.unique(proteins, axis = 0)
              prot_count = len(unique_proteins)
              bag = (unique_proteins, prot_count)
            if count == len(data) - 1:
              return bag

# NEGATIVES

def shifting_window_negatives(bag, nterm, seq_length_left , seq_length_right , aa_left_N = 2, aa_right_C = 2, number_of_negatives = 1):

      #bag[0]: sequence
      labeled_sequences = []
      sequences = []
      neg_sequences = []
      x = 0
      epitope_length = len(bag[0][0]) - (seq_length_left + seq_length_right + aa_left_N + aa_right_C)

      while x <= epitope_length + aa_right_C + aa_left_N -1 :
            
            sequences = []            
            sequence = ''
            b = seq_length_left + x 
            start = b - seq_length_left
            end = b + seq_length_right 
            slicing = slice(start, end)
            for i in bag[0]:  
              sequence  = sequence + i[slicing]
          
              #classification  
              #n-terminal
            if x==aa_left_N and nterm > 0:
                sequences.append(sequence)
                classification = 1
                labeled_sequences.append((sequences[-1], classification, bag[1]))
              #c-terminal
            if x== epitope_length + aa_left_N:
                sequences.append(sequence)
                classification = 1
                labeled_sequences.append((sequences[-1], classification, bag[1]))
            
            if x != aa_left_N and x != epitope_length + aa_left_N:
                neg_sequences.append(sequence)

            if x == epitope_length + aa_right_C + aa_left_N -1:
                for seq in random.sample(neg_sequences, number_of_negatives):
                    sequences.append(seq)
                    classification = 0
                    labeled_sequences.append((sequences[-1], classification, bag[1]))

            x = x + 1
            

      return labeled_sequences
      
def shuffled_negatives(bag, number_of_negatives,nterm,seq_length_left , seq_length_right , aa_left_N = 2, aa_right_C = 2):

      #bag[0]: sequence
      labeled_sequences = []
      sequences = []
      neg_sequences = []
      x = 0
      epitope_length = len(bag[0][0]) - (seq_length_left + seq_length_right + aa_left_N + aa_right_C)
      
      for x in [aa_left_N, epitope_length + aa_left_N]:
            sequences = []            
            sequence = ''
            neg_seq_chain = ''
            b = seq_length_left + x 
            start = b - seq_length_left
            end = b + seq_length_right 
            slicing = slice(start, end)
            for i in bag[0]:  
              sequence  = sequence + i[slicing]
          
              #classification  
              #n-terminal
            
            if x==aa_left_N and nterm > 0:
                labeled_sequences.append((sequence, 1, bag[1]))
                neg_seq = ''.join(random.sample(sequence,len(sequence)))
                labeled_sequences.append((neg_seq, 0, bag[1]))
              #c-terminal
            if x== epitope_length + aa_left_N:
                labeled_sequences.append((sequence, 1, bag[1]))
                for c in range(bag[1]):
                        pos_seq = sequence[c*(seq_length_left + seq_length_right):(c+1)*(seq_length_left + seq_length_right)]
                    
                        if 'Z' not in pos_seq:
                                neg_seq  = ''.join(random.sample(pos_seq,len(pos_seq)))
                                neg_seq_chain = neg_seq_chain + neg_seq
                        else: 
                                for z in range(seq_length_left+ seq_length_right):
                                        if z == 0 and pos_seq[z] == 'Z' and pos_seq[z+1] != 'Z':
                                                neg_seq = 'Z' + ''.join(random.sample(pos_seq[1:],len(pos_seq[1:])))
                                                neg_seq_chain = neg_seq_chain + neg_seq
                                                break                        
                                        if 1<z<seq_length_left and pos_seq[z] != 'Z' and pos_seq[z-1] == 'Z': 
                                                neg_seq = z * 'Z' + ''.join(random.sample(pos_seq[z:],len(pos_seq[z:])))
                                                neg_seq_chain = neg_seq_chain + neg_seq
                                                break
                                        if z >= seq_length_left and pos_seq[z] == 'Z' and pos_seq[z-1] != 'Z':
                                                neg_seq = ''.join(random.sample(pos_seq[:z],len(pos_seq[:z]))) + (seq_length_left+ seq_length_right - z) * 'Z'
                                                neg_seq_chain = neg_seq_chain + neg_seq
                                                break
                
                labeled_sequences.append((neg_seq_chain, 0, bag[1]))
                neg_seq_chain = ''
            
            if x != aa_left_N and x != epitope_length + aa_left_N:
                neg_sequences.append(sequence)

            if x == epitope_length + aa_right_C + aa_left_N -1:
                for seq in random.sample(neg_sequences, number_of_negatives):
                    sequences.append(seq)
                    classification = 0
                    labeled_sequences.append((sequences[-1], classification, bag[1]))

            x = x + 1
            

      return labeled_sequences
      

# LABELED BAGS
     
def label_bags(data,negatives_generator, number_of_negatives = 1, nterm = 0, seq_length_left = 15, seq_length_right = 15):
    
    bags = generate_bags(negatives_generator, data, nterm,seq_length_left = seq_length_left, seq_length_right = seq_length_right)    
    labeled_bags = []

    if negatives_generator == 'shifting_window_negatives':         
            for bag in bags:            
                labeled_bag = shifting_window_negatives(bag=bag, number_of_negatives = number_of_negatives+ number_of_negatives*nterm, nterm=nterm, seq_length_left = seq_length_left, seq_length_right = seq_length_right)              
                for i in range(len(labeled_bag)):
                    labeled_bags.append(labeled_bag[i])

    if negatives_generator == 'shuffled_negatives':
            for bag in bags:
                labeled_bag = shuffled_negatives(bag=bag, number_of_negatives = number_of_negatives+ number_of_negatives*nterm, nterm=nterm, seq_length_left = seq_length_left, seq_length_right = seq_length_right)              
                for i in range(len(labeled_bag)):
                    labeled_bags.append(labeled_bag[i])

    df = pd.DataFrame(data = labeled_bags)
    df = df.rename(columns={0: "proteins", 1: "label", 2: "count"})
    return df

