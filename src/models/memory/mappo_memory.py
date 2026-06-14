# src/models/memory/mappo_memory.py

from .memory import Memory
import torch


class MAPPOMemory(Memory):
    """
    Buffer mémoire pour MAPPO.
    """

    def __init__(self, batch_size=2500):
        super().__init__(batch_size)

        self.local_observations = []
        self.global_observations = []

        self.actions = []

        self.values = []

        self.log_probs = []

        self.rewards = []

        self.terminals = []

    def remember(
        self,
        local_observation,
        global_observation,
        action,
        value,
        log_prob,
        reward,
        terminal,
    ):
        """
        Ajoute une transition MAPPO dans la mémoire.
        """

        self.local_observations.append(local_observation)
        self.global_observations.append(global_observation)

        self.actions.append(action)

        self.values.append(value)

        self.log_probs.append(log_prob)

        self.rewards.append(reward)

        self.terminals.append(terminal)

    def clear(self):
        """
        Vide complètement le buffer mémoire.
        """

        self.local_observations.clear()
        self.global_observations.clear()

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
            local_observations, global_observations, actions, values, log_probs, rewards, terminals, batch_idxs
        '''
        idxs = torch.randperm(len(self.actions))
        batch_idxs = idxs.split(self.batch_size)

        return self.local_observations, self.global_observations, self.actions, self.values, \
            self.log_probs, self.rewards, self.terminals, batch_idxs

    def __len__(self):
        """
        Retourne le nombre de transitions stockées.
        """

        return len(self.actions)