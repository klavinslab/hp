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

    def __init__(self, capacity,dev="cpu"):
        self.capacity = capacity
        self.dev = dev
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.next_states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = range(len(self))
        subset      = np.random.choice(indices,size=batch_size,replace=False).tolist()     
        states      = torch.stack([x for x in Batch(subset,self.states)])
        next_states = torch.stack([x for x in Batch(subset,self.next_states)])
        actions     = torch.stack([x for x in Batch(subset,self.actions)])        
        rewards     = torch.tensor([x for x in Batch(subset,self.rewards)], dtype=torch.float)
        rewards = rewards.to(self.dev)
        return states, next_states, actions, rewards

    def __len__(self):
        return len(self.states)