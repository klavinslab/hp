import torch
import torch.nn as nn
import torch.nn.functional as F

class Energy(nn.Module)   :
    
    def __init__(self, n):
        super(Energy,self).__init__()
        self.n = n
        self.fc1 = nn.Linear(2*self.n + 3*(self.n-1),256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.relu(self.fc1(x.view(-1,2*self.n + 3*(self.n-1))))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def combine(self,sequences,conformations):
        s = sequences.view(-1,2*self.n)
        c = conformations.view(-1,3*(self.n-1))
        return torch.cat([s,c],dim=1)