from collections import OrderedDict

import CybORG.Shared.Enums as Enums

from .base import Node

class SystemNode(Node):
    '''
    Node representing computers (servers, users, and routers)
    '''
    def __init__(self, uuid: int, observation: dict, is_server=False, is_router=False, crown_jewel=False):
        self.feats = OrderedDict(
            Architecture = None,
            OSDistro = None,
            OSType = None,
            OSVersion = None,
            OSKernelVersion = None,
            os_patches = [],
            crown_jewel=float(crown_jewel),
            user=float(not is_server),
            server=float(is_server),
            router=float(is_router)
        )

        self.dims = [
            len(Enums.Architecture.__members__),
            len(Enums.OperatingSystemDistribution.__members__),
            len(Enums.OperatingSystemType.__members__),
            len(Enums.OperatingSystemVersion.__members__),
            len(Enums.OperatingSystemKernelVersion.__members__),
            len(Enums.OperatingSystemPatch.__members__),
            1,1,1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.Architecture.__members__.items()) +
            list(Enums.OperatingSystemDistribution.__members__.items()) +
            list(Enums.OperatingSystemType.__members__.items()) +
            list(Enums.OperatingSystemVersion.__members__.items()) +
            list(Enums.OperatingSystemKernelVersion.__members__.items()) +
            list(Enums.OperatingSystemPatch.__members__.items()) +
            ["crown_jewel", "user", "server", "router"]
        )
