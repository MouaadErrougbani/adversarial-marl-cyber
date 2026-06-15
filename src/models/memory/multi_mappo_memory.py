# src/models/memory/multi_mappo_memory.py

from .multi_memory import MultiMemory
from .mappo_memory import MAPPOMemory
import torch

class MultiMAPPOMemory(MultiMemory):
    """
    Gère une mémoire MAPPO pour chaque agent.
    """

    def __init__(
        self,
        batch_size,
        num_agents=5,
    ):
        super().__init__(
            batch_size,
            num_agents,
        )

        self.memories = [
            MAPPOMemory(
                self.batch_size
            )
            for _ in range(
                self.num_agents
            )
        ]
    def remember(
        self,
        agent_idx,
        *args,
    ):
        """
        Ajoute une transition dans la mémoire de l'agent spécifié.
        """

        self.memories[agent_idx].remember(
            *args
        )

    def clear(self):
        """
        Vide toutes les mémoires des agents.
        """

        for memory in self.memories:
            memory.clear()

    def __len__(self):
        """
        Retourne le nombre total de transitions stockées.
        """

        return sum(
            len(memory)
            for memory in self.memories
        )
    
    def get_batches(self):
        """
        Fusionne les mémoires de tous les agents et
        retourne les données sous forme de mini-batchs
        avec un shuffle global.
        """
        offset = 0
        batch_indices = []

        all_local_observations = []
        all_global_observations = []

        all_actions = []
        all_values = []

        all_log_probs = []

        all_rewards = []
        all_terminals = []

        # Fusion de toutes les mémoires
        for memory in self.memories:

            all_local_observations += (
                memory.local_observations
            )

            all_global_observations += (
                memory.global_observations
            )

            all_actions += memory.actions
            all_values += memory.values

            all_log_probs += memory.log_probs

            all_rewards += memory.rewards
            all_terminals += memory.terminals

            # cnt = len(memory.local_observations)

            # idx = torch.randperm(cnt) + offset 
            # batch_indices += list(idx.split(self.batch_size))
            # offset += cnt

        # Nombre total de samples
        total_samples = len(all_actions)

        # Shuffle global
        global_indices = torch.randperm(
            total_samples
        )

        # Mini-batches globaux
        batch_indices = global_indices.split(
            self.batch_size
        )

        return (
            all_local_observations,
            all_global_observations,
            all_actions,
            all_values,
            all_log_probs,
            all_rewards,
            all_terminals,
            batch_indices,
        )

