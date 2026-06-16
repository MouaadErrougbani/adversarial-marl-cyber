# src/models/memory/multi_maddpg_memory.py

from .multi_memory import MultiMemory
from .maddpg_memory import MADDPGMemory

import torch


class MultiMADDPGMemory(MultiMemory):
    """
    Multi-agent replay buffer pour MADDPG.
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
            MADDPGMemory(
                self.batch_size
            )
            for _ in range(
                self.num_agents
            )
        ]

    def remember(
        self,
        agent_idx,
        *args
    ):
        """
        Stocke une transition pour un agent.
        """

        self.memories[
            agent_idx
        ].remember(
            *args
        )

    def clear(self):
        """
        Vide complètement la mémoire.
        """

        for memory in self.memories:

            memory.clear()

    def __len__(self):
        """
        Retourne le nombre total
        de transitions stockées.
        """

        return sum(

            len(memory)

            for memory in self.memories

        )

    def get_batches(self):
        """
        Retourne toutes les données nécessaires
        à l'entraînement MADDPG.
        """

        offset = 0

        batch_indices = []

        all_local_observations = []

        all_global_observations = []

        all_actions = []

        all_rewards = []

        all_next_local_observations = []

        all_next_global_observations = []

        all_joint_next_local_observations = []

        all_terminals = []

        all_joint_actions = []

        for memory in self.memories:

            all_local_observations += (
                memory.local_observations
            )

            all_global_observations += (
                memory.global_observations
            )

            all_actions += (
                memory.actions
            )

            all_rewards += (
                memory.rewards
            )

            all_next_local_observations += (
                memory.next_local_observations
            )

            all_next_global_observations += (
                memory.next_global_observations
            )

            all_joint_next_local_observations += (
                memory.joint_next_local_observations
            )

            all_terminals += (
                memory.terminals
            )

            all_joint_actions += (
                memory.joint_actions
            )

            cnt = len(
                memory.local_observations
            )

            idx = (
                torch.randperm(cnt)
                + offset
            )

            batch_indices += list(
                idx.split(
                    self.batch_size
                )
            )

            offset += cnt

        return (

            all_local_observations,

            all_global_observations,

            all_actions,

            all_rewards,

            all_next_local_observations,

            all_next_global_observations,

            all_joint_next_local_observations,

            all_terminals,

            all_joint_actions,

            batch_indices,
        )