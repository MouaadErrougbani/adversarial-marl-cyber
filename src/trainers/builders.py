# src/trainers/builders.py

from src import (
    make_env,
    InductiveGraphPPOAgent,
    InductiveGraphMAPPOAgent,
    InductiveGraphMADDPGAgent,
    ObservationGraph,
)


def build_agents(cfg):
    """
    Build agents.
    """
    algorithm = cfg["train"]["algorithm"].upper()
    agent_registry = {
        "PPO": InductiveGraphPPOAgent,
        "MAPPO" : InductiveGraphMAPPOAgent,
        "MADDPG": InductiveGraphMADDPGAgent
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
                        **cfg.get("hyperparams", {})
                    )
                    shared_critic = agent.critic
                agents.append(agent)
            return agents
        
        elif algorithm == "MADDPG":

            agents = [

                AgentClass(
                    in_dim=ObservationGraph.DIM + 5,
                    a_kwargs=cfg.get("actor", {}),
                    c_kwargs=cfg.get("critic", {}),
                    bs=cfg["train"]["batch_size"],
                    num_agents=num_agents,
                    **cfg.get("hyperparams", {})
                )

                for _ in range(num_agents)
            ]

            #
            # Chaque agent doit connaître
            # les autres agents
            #

            for idx, agent in enumerate(
                agents
            ):

                agent.agent_id = idx

                agent.set_agents(
                    agents
                )
                
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