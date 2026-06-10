# Définition de l'espace d'actions CAGE4

from CybORG.Simulator.Actions import (
    Analyse,
    Remove,
    Restore,
    DeployDecoy,
    Monitor,
)

from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)

from .constants import MAX_HOSTS
from .topology import ROUTERS

NODE_ACTIONS = [
    Analyse,
    Remove,
    Restore,
    DeployDecoy,
]

EDGE_ACTIONS = [
    AllowTrafficZone,
    BlockTrafficZone,
]

GLOBAL_ACTIONS = [
    Monitor,
]

N_NODE_ACTIONS = len(NODE_ACTIONS)
N_EDGE_ACTIONS = len(EDGE_ACTIONS)
N_GLOBAL_ACTIONS = len(GLOBAL_ACTIONS)

MAX_ACTIONS = (
    N_NODE_ACTIONS * MAX_HOSTS
    + N_EDGE_ACTIONS * (len(ROUTERS) - 1)
    + N_GLOBAL_ACTIONS
)