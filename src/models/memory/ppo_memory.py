# src/models/memory/ppo_memory.py

import torch 
from .memory import Memory

class PPOMemory(Memory):
    '''
    Holds memories for agents that are relevant to the 
    PPO optimization procedure
    '''
    def __init__(self, batch_size=2500):
        super().__init__(batch_size)
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.terminals = []



    def remember(self, state, action, value, log_prob, reward, terminal):
        '''
        Ajoute une nouvelle expérience dans la mémoire. 

        Args:
            state: State
            action: Action
            value: Value (critic output)
            log_prob: Log Prob (actor output)
            reward: Reward
            terminal: Terminal 
        '''
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def clear(self): 
        '''
        Vide complètement le buffer mémoire. 
        '''
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.terminals.clear()


    def get_batches(self):
        '''
        Return chunks of the shuffled memory buffer 
        randomly partitioned into `self.batch_size`-sized chunks 
        Returns:
            states, actions, values, log_probs, rewards, terminals, batch_idxs
        '''
        idxs = torch.randperm(len(self.actions))
        batch_idxs = idxs.split(self.batch_size)

        return self.states, self.actions, self.values, \
            self.log_probs, self.rewards, self.terminals, batch_idxs

    def __len__(self):
        """
        Retourne le nombre d'expériences stockées.
        Returns:
            int: Le nombre d'expériences actuellement stockées dans la mémoire.
        """
        return len(self.actions)

