# src/trainers/builders.py

from src import (
    make_env,
    InductiveGraphPPOAgent,
    InductiveGraphMAPPOAgent,
    ObservationGraph,
)


def build_agents(cfg, device = "cpu"):
    """
    Build agents.
    """
    algorithm = cfg["train"]["algorithm"].upper()
    agent_registry = {
        "PPO": InductiveGraphPPOAgent,
        "MAPPO" : InductiveGraphMAPPOAgent
    }

    num_agents = cfg["train"]["num_agents"]
    if algorithm not in agent_registry:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    AgentClass = agent_registry[algorithm]

    def genereate_agents():
        if algorithm == "PPO":
            return [
                AgentClass(
                    in_dim=ObservationGraph.DIM + 5,
                    a_kwargs=cfg.get("actor", {}),
                    c_kwargs=cfg.get("critic", {}),
                    bs=cfg["train"]["batch_size"],
                    epochs=cfg["train"]["epochs"],
                    num_agents=num_agents,
                    device=device,
                    **cfg.get("hyperparams", {})
                )
                for _ in range(num_agents)
            ]
        
        elif algorithm == "MAPPO":
            agents = []
            shared_critic = None
            for _ in range(num_agents):
                if shared_critic is not None:
                    agent = AgentClass(
                        in_dim=ObservationGraph.DIM + 5,
                        a_kwargs=cfg.get("actor", {}),
                        c_kwargs=cfg.get("critic", {}),
                        bs=cfg["train"]["batch_size"],
                        epochs=cfg["train"]["epochs"],
                        num_agents=num_agents,
                        device=device,
                        critic=shared_critic,
                        **cfg.get("hyperparams", {})
                    )
                    
                else:
                    agent = AgentClass(
                        in_dim=ObservationGraph.DIM + 5,
                        a_kwargs=cfg.get("actor", {}),
                        c_kwargs=cfg.get("critic", {}),
                        bs=cfg["train"]["batch_size"],
                        epochs=cfg["train"]["epochs"],
                        num_agents=num_agents,
                        device=device,
                        **cfg.get("hyperparams", {})
                    )
                    shared_critic = agent.critic
                agents.append(agent)
            return agents
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    agents = genereate_agents()

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