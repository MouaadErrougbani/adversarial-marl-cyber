from collections import OrderedDict

import CybORG.Shared.Enums as Enums

from .base import Node


class FileNode(Node):
    '''
    Nodes representing files on workstations. 
    These are almost always malicious. 
    '''
    def __init__(self, uuid: int, observation: dict, is_new=True):
        self.feats = OrderedDict(
            [
                (s, None) for s in [
                    'Known File',
                    'Known Path',
                    'User Permissions',
                    'Group Permissions',
                    'Default Permissions'
                ]
            ],
            Version=None,
            Type=None,
            Vendor=None,

            # Additional static features
            is_new=float(is_new),

            # Will be updated if observed later
            Density= -1.,
            Signed= -1.
        )

        self.dims = [
            len(Enums.FileType.__members__),
            len(Enums.Path.__members__),
            8,8,8, # Permissions groups
            len(Enums.FileVersion.__members__),
            len(Enums.FileType.__members__),
            len(Enums.Vendor.__members__),
            1,
            1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.FileType.__members__.items()) +
            list(Enums.Path.__members__.items()) +
            ['permissions']*24 +
            list(Enums.FileVersion.__members__.items()) +
            list(Enums.FileType.__members__.items()) +
            list(Enums.Vendor.__members__.items()) +
            ['is new', 'density', 'signed']
        )
