import torch
import torch.nn as nn
import torch.nn.functional as F

class Translate(nn.Module):
    
    def __init__(self,encoder,decoder):
        super(Translate,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sequence):
        return self.decode(self.encode(sequence))
    
    def encode(self,sequence):
        return self.encoder(sequence)
    
    def decode(self,code):
        return self.decoder(code)
    
class Encoder(nn.Module):
    
    def __init__(self,n):
        super(Encoder,self).__init__()
        self.n = n
        self.fc1 = nn.Linear(2*self.n,100)
        self.fc2 = nn.Linear(100,100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self,x):
        x = x.view(-1,2*self.n)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, n, temperature=1):
        super(Decoder,self).__init__()
        self.temperature = temperature
        self.n = n
        self.fc1 = nn.Linear(100,100)
        self.fc2 = nn.Linear(100,3*(self.n-1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1,self.n-1,3)
        x /= self.temperature
        x = F.softmax(x,dim=2)
        return x
