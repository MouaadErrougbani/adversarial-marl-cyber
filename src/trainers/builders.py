# src/trainers/builders.py

from src import (
    make_env,
    InductiveGraphPPOAgent,
    ObservationGraph,
)


def build_agents(cfg, device = "cpu"):
    """
    Build agents.
    """
    algorithm = cfg["train"]["algorithm"]
    agent_registry = {
        "ppo": InductiveGraphPPOAgent,
    }

    agent_count = cfg["train"]["agent_count"]
    if algorithm not in agent_registry:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    AgentClass = agent_registry[algorithm]

    agents = [
        AgentClass(
            in_dim=ObservationGraph.DIM + 5,
            a_kwargs=cfg.get("actor", {}),
            c_kwargs=cfg.get("critic", {}),
            bs=cfg["train"]["batch_size"],
            epochs=cfg["train"]["epochs"],
            agent_count=agent_count,
            device=device,
            **cfg.get("hyperparams", {})
        )
        for _ in range(agent_count)
    ]

    return agents

def build_envs(cfg):
    """
    Build rollout environments.
    """

    workers = cfg["train"]["workers"]

    episode_len = (
        cfg["train"]["episode_len"]
    )

    seed = cfg["train"]["seed"]

    envs = [
        make_env(
            seed=seed + i,
            steps=episode_len,
        )
        for i in range(workers)
    ]

    return envs