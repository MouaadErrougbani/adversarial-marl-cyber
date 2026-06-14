# src/models/memory/multi_memory.py

from abc import ABC, abstractmethod


class MultiMemory(ABC):
    """
    Base abstract class for all multi-agent memory buffers.
    """

    def __init__(self, batch_size, num_agents):
        # Taille des mini-batchs utilisés pendant l'entraînement
        self.batch_size = batch_size

        # Nombre total d'agents
        self.num_agents = num_agents

    @abstractmethod
    def remember(self, *args, **kwargs):
        """
        Ajouter une expérience dans la mémoire d'un agent.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Vider complètement toutes les mémoires.
        """
        pass

    @abstractmethod
    def get_batches(self):
        """
        Retourner les données sous forme de mini-batchs.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Retourner le nombre total d'expériences stockées.
        """
        pass