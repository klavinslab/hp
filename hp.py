import random
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class HP():
    
    def __init__(self,sequence,conformation=None):
        
        self.sequence = sequence
        
        if conformation is None:
            self.conformation = HP.initial_conformation(len(self))
        else:
            self.conformation = conformation
            
    def __len__(self):
        return len(self.sequence)

    @classmethod       
    def initial_conformation(cls,n):
        return "".join(["S" for _ in range(n-1)])
    
    @classmethod       
    def symbol_to_angle(cls,s):
        if s == "S":
            return 0
        elif s == "R":
            return math.pi / 2
        elif s == "L":
            return -math.pi / 2
        else:
            raise Exception("Unknown symbol '%s' in conformation" % s)
            
    @classmethod
    def random(cls,n,h_prob=0.5):
        return cls("".join(["H" if random.random() < h_prob else "P" for _ in range(n)]))
    
    def randomize_conf(self):
        a = []
        for i in range(len(self)-1):
            r = random.random()
            if r < 1/3:
                a.append("R")
            elif r < 2/3:
                a.append("L")
            else:
                a.append("S")
        return "".join(a)
    
    def one_hot(self,dev="cpu"):
        return torch.tensor([[1,0] if s == "H" else [0,1] for s in self.sequence],dtype=torch.float,device=dev)

    def conf_one_hot(self,dev="cpu"):
        return torch.tensor([
            [1,0,0] if s == "R" 
                    else [0,1,0] if s == "L" 
                    else [0,0,1]
                    for s in self.conformation],dtype=torch.float,device=dev)    
   
    def make_state(self, dev="cpu"):
        s = self.one_hot(dev=dev)
        c = self.conf_one_hot(dev=dev)
        z = torch.zeros(1,3)
        z = z.to(dev)
        c = torch.cat([c,z])
        return torch.cat([s,c],1)

    def coordinates(self, conformation=None):
        if conformation is None:
            c = self.conformation
        else:
            c = conformation        
        x = [0]
        y = [0]
        a = 0
        for i in range(len(self)):
            if i < len(c):
                a += HP.symbol_to_angle(c[i])
                x.append(x[i]+int(math.cos(a)))
                y.append(y[i]+int(math.sin(a)))
        return x, y

    def show(self,ax):
        ax.set_aspect(1)
        x,y = self.coordinates()
        for i in range(len(self)):
            color = 'gray' if self.sequence[i] == 'H' else 'white'
            c = plt.Circle((x[i],y[i]), 0.25, edgecolor='black', facecolor=color)
            ax.add_artist(c)
            if i < len(self)-1:
                ax.plot([x[i],x[i+1]],[y[i],y[i+1]],zorder=-1,c='black')
        ax.set_xlim(min(x)-5,max(x)+5)
        ax.set_ylim(min(y)-5,max(y)+5)
    
    @classmethod   
    def dist(cls,x1,x2,y1,y2):
        return math.sqrt(((x1-x2)**2 + (y1-y2)**2))
    
    def energy(self,conformation=None):
        if conformation is None:
            c = self.conformation
        else:
            c = conformation
        x,y=self.coordinates(c)
        e = 0
        for i in range(len(self)):
            for j in range(i):
                if HP.dist(x[i],x[j],y[i],y[j]) <= 0.1:
                    e += 100
                if i > j + 1 \
                    and self.sequence[i] == 'H' \
                    and self.sequence[j] == 'H' \
                    and HP.dist(x[i],x[j],y[i],y[j]) <= 1.01:
                    e -= 1
        return e

    def random_move(self):
        i = random.randint(0,len(self)-2)
        m = self.conformation[i]
        while m == self.conformation[i]:
            m = ["R","L","S"][random.randint(0,2)]
        cnew = self.conformation
        cnew = list(cnew)
        cnew[i] = m
        cnew = "".join(cnew)
        return cnew

    def minimize(self, n=100, t=1, dt=0.1):
        best = self.conformation
        e = self.energy()
        ebest = e
        elist = [e]
        clist = [self.conformation]
        for m in range(n):
            cnew = self.random_move()
            enew = self.energy(conformation=cnew)
            alpha = dt * math.exp(-(enew-e)/t)
            if random.random() <= alpha:
                self.conformation = cnew
                e = enew               
                elist.append(e)
                clist.append(self.conformation)
                if e <= ebest:
                    best = self.conformation
                    ebest = e
        self.conformation = best
        return best, elist, clist
    
    def minimize_step(self,t=1,dt=0.1):
        e = self.energy()
        cnew = self.random_move()
        enew = self.energy(conformation=cnew) 
        alpha = dt * math.exp(-(enew-e)/t)
        while random.random() > alpha:
            e = self.energy()
            cnew = self.random_move()
            enew = self.energy(conformation=cnew) 
            alpha = dt * math.exp(-(enew-e)/t) 
        self.conformation = cnew
        return enew

    @classmethod
    def one_hot_seq_to_string(cls,seq_oh):
        return "".join(["H" if s[0] == 1 else "P" for s in seq_oh])
    
    @classmethod
    def sample_from_conf_one_hot(cls,conf_oh):
        x = conf_oh.cpu().detach().numpy()
        return "".join([np.random.choice(["R","L","S"],p=p) for p in x])
    
    @classmethod
    def energy_of_one_hot(cls,sequence,conformation):
        return HP.create_from_one_hots(sequence,conformation).energy()
    
    @classmethod
    def create_from_one_hots(cls, seq, conf):
        s = HP.one_hot_seq_to_string(seq)
        c = HP.sample_from_conf_one_hot(conf)
        return HP(s,c)        
        
