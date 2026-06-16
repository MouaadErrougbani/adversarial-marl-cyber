# src/models/memory/maddpg_memory.py

from .memory import Memory
import torch


class MADDPGMemory(Memory):
    """
    Replay buffer local pour un agent MADDPG.
    """

    def __init__(self, batch_size=1024):
        super().__init__(batch_size)

        self.local_observations = []

        self.actions = []

        self.rewards = []

        self.next_local_observations = []

        self.terminals = []

        self.global_observations = []

        self.next_global_observations = []

        self.joint_actions = []

        self.joint_next_local_observations = []

    def remember(
        self,
        local_observation,
        global_observation,
        action,
        reward,
        next_local_observation,
        next_global_observation,
        joint_next_local_observations,
        terminal,
        joint_action,
    ):
        """
        Ajoute une transition dans la mémoire.
        """

        self.local_observations.append(
            local_observation
        )

        self.global_observations.append(
            global_observation
        )

        self.actions.append(
            action
        )

        self.rewards.append(
            reward
        )

        self.next_local_observations.append(
            next_local_observation
        )

        self.next_global_observations.append(
            next_global_observation
        )

        self.joint_next_local_observations.append(
            joint_next_local_observations
        )

        self.terminals.append(
            terminal
        )

        self.joint_actions.append(
            joint_action
        )

    def clear(self):
        """
        Vide complètement le buffer.
        """

        self.local_observations.clear()

        self.global_observations.clear()

        self.actions.clear()

        self.rewards.clear()

        self.next_local_observations.clear()

        self.next_global_observations.clear()

        self.joint_next_local_observations.clear()

        self.terminals.clear()

        self.joint_actions.clear()

    def get_batches(self):
        """
        Retourne les données avec des mini-batchs
        aléatoires compatibles avec le framework actuel.
        """

        idxs = torch.randperm(
            len(self.actions)
        )

        batch_idxs = idxs.split(
            self.batch_size
        )

        return (
            self.local_observations,
            self.global_observations,

            self.actions,
            self.rewards,

            self.next_local_observations,
            self.next_global_observations,

            self.joint_next_local_observations,

            self.terminals,

            self.joint_actions,

            batch_idxs,
        )

    def __len__(self):
        """
        Nombre de transitions stockées.
        """

        return len(
            self.actions
        )