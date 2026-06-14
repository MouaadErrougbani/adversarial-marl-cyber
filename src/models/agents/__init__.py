# src/models/agents/__init__.py

from .ppo_agent import (
    InductiveGraphPPOAgent,
    load,
)

from .mappo_agent import (
    InductiveGraphMAPPOAgent,
)

__all__ = [
    "InductiveGraphPPOAgent",
    "InductiveGraphMAPPOAgent",
    "load",
]