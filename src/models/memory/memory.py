# src/models/memory/memory.py

from abc import ABC, abstractmethod


class Memory(ABC):
    """
    Base abstract class for all memory buffers.
    """

    def __init__(self, batch_size):
        # Taille des mini-batchs utilisés pendant l'entraînement
        self.batch_size = batch_size

    @abstractmethod
    def remember(self, *args, **kwargs):
        """
        Ajouter une expérience dans la mémoire.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Vider complètement la mémoire.
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
        Retourner le nombre d'expériences stockées.
        """
        pass

