# src/models/memory/multi_ppo_memory.py
import torch

from .ppo_memory import PPOMemory
from .multi_memory import MultiMemory

class MultiPPOMemory(MultiMemory):
    '''
    Store multiple memory buffers, one for each agent. 
    Used during training to keep agent's observations seperated 
    '''
    def __init__(self, batch_size, num_agents=5) -> None:
        super().__init__(batch_size, num_agents)
        self.memories = [PPOMemory(self.batch_size) for _ in range(self.num_agents)]

    def remember(self, idx_agent, *args):
        """
            Ajoute une expérience dans la mémoire de l'agent spécifié.

            Args:
                idx_agent (int): Indice de l'agent.
                *args: Données de transition PPO.
        """
        self.memories[idx_agent].remember(*args)

    def clear(self):
        """
            Vide toutes les mémoires des agents.
        """
        for mem in self.memories:
            mem.clear()
        
    def get_batches(self):
        all_states = []
        all_actions = []
        all_values = []
        all_log_probs = []
        all_rewards = []
        all_terminals = []

        # 1) Fusion de tous les samples
        for memory in self.memories:
            all_states += memory.states
            all_actions += memory.actions
            all_values += memory.values
            all_log_probs += memory.log_probs
            all_rewards += memory.rewards
            all_terminals += memory.terminals

        # 2) Nombre total de samples
        total_samples = len(all_actions)

        # 3) Shuffle global
        global_indices = torch.randperm(total_samples)

        # 4) Mini-batches globaux
        batch_indices = global_indices.split(self.batch_size)
        return (
            all_states,
            all_actions,
            all_values,
            all_log_probs,
            all_rewards,
            all_terminals,
            batch_indices,
        )
    
    def __len__(self):
        """
        Retourne le nombre total d'expériences stockées.
        """
        return sum(len(mem) for mem in self.memories)
    