import numpy as np
import torch
import sys
import random

class Batch:
    
  def __init__(self, indices, source):
    self.indices = indices
    self.source = source    
    self.i = 0
    self.n = len(indices)    
    
  def __iter__(self):
    self.i = 0
    return self

  def __next__(self):
    if self.i < self.n:
        x = self.source[self.indices[self.i]]
        self.i += 1
        return x
    else:
        raise StopIteration
        
class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.sequences = []
        self.conformations = []
        self.position = 0

    def push(self, sequence, conformation):
        if len(self.sequences) < self.capacity:
            self.sequences.append(None)
            self.conformations.append(100)
        self.sequences[self.position] = sequence
        self.conformations[self.position] = conformation
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, dev="cpu"):
        indices = range(len(self))
        subset = np.random.choice(indices,size=batch_size,replace=False).tolist()     
        seqs  = torch.stack([x for x in Batch(subset,self.sequences)])
        confs = torch.stack([x for x in Batch(subset,self.conformations)])
        seqs = seqs.to(dev)
        confs = confs.to(dev)
        return subset, seqs, confs

    def __len__(self):
        return len(self.sequences)