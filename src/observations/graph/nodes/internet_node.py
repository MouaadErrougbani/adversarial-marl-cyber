from collections import OrderedDict

from .base import Node

class InternetNode(Node):
    '''
    No features. Purely a structural node
    '''
    def __init__(self, uuid: int, observation: dict=dict()):
        self.feats = OrderedDict()
        self.dims = []
        self.dim = 0

        super().__init__(uuid, observation)

    @property
    def labels(self):
        return []
