from collections import OrderedDict

import CybORG.Shared.Enums as Enums

from .base import Node


class ConnectionNode(Node):
    '''
    Node representing processes that communicate with other hosts 
    '''
    def __init__(self, uuid: int, observation: dict=dict(), suspicious_pid: bool=False, is_decoy: bool=False, is_default=False, is_ephemeral=False):
        self.feats = OrderedDict(
            process_name = None,
            process_type = None,
            suspicious_pid=float(suspicious_pid),
            is_decoy=float(is_decoy),
            is_default=float(is_default),
            is_ephemeral=float(is_ephemeral)
        )
        self.dims = [
            len(Enums.ProcessName.__members__),
            len(Enums.ProcessType.__members__),
            1,1,1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.ProcessName.__members__.items()) +
            list(Enums.ProcessType.__members__.items()) +
            ['suspicious pid', 'is decoy', 'is default', 'is_ephemeral']
        )


