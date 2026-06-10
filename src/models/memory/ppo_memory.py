# src/models/memory/ppo_memory.py

import torch 

class PPOMemory:
    '''
    Holds memories for agents that are relevant to the 
    PPO optimization procedure
    '''
    def __init__(self, bs):
        self.s = []
        self.a = []
        self.v = []
        self.p = []
        self.r = []
        self.t = []

        self.bs = bs 

    def remember(self, s,a,v,p,r,t):
        '''
        Pushes new memory into the buffer 

        Args:
            s: State
            a: Action
            v: Value (critic output)
            p: Log Prob (actor output)
            r: Reward
            t: Terminal 
        '''
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r) 
        self.t.append(t)

    def clear(self): 
        '''
        Empties the memory buffer 
        '''
        self.s = []; self.a = []
        self.v = []; self.p = []
        self.r = []; self.t = []

    def get_batches(self):
        '''
        Return chunks of the shuffled memory buffer 
        randomly partitioned into `self.bs`-sized chunks 
        '''
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs
