from pathlib import Path

import torch

from src import (
    InductiveGraphPPOAgent,
    InductiveGraphMAPPOAgent,
    InductiveGraphMADDPGAgent,
)


def build_agents(cfg):
    """
    Build evaluation agents and load their trained checkpoints.
    """

    # Récupérer la configuration d'évaluation
    eval_cfg = cfg["eval"]

    # Récupérer l'algorithme utilisé
    algorithm = eval_cfg["algorithm"].upper()

    # Associer chaque algorithme à sa classe
    agent_registry = {
        "PPO": InductiveGraphPPOAgent,
        "MAPPO": InductiveGraphMAPPOAgent,
        "MADDPG": InductiveGraphMADDPGAgent,
    }

    # Vérifier que l'algorithme est supporté
    if algorithm not in agent_registry:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    AgentClass = agent_registry[algorithm]

    # Récupérer les paramètres d'évaluation
    num_agents = eval_cfg["num_agents"]
    checkpoint_dir = Path(eval_cfg["checkpoint_dir"])
    checkpoint_pattern = eval_cfg["checkpoint_pattern"]
    device = eval_cfg.get("device", "cpu")

    # Vérifier que le dossier des checkpoints existe
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    agents = []
    shared_critic = None

    for agent_id in range(num_agents):
        # Construire le nom du checkpoint
        checkpoint_name = checkpoint_pattern.format(
            algorithm=algorithm.lower(),
            agent_id=agent_id,
        )

        checkpoint_path = checkpoint_dir / checkpoint_name

        # Vérifier que le checkpoint existe
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for agent {agent_id}: "
                f"{checkpoint_path}"
            )

        # Charger le checkpoint sur le CPU
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        # Vérifier la présence des données nécessaires
        required_keys = {"agent", "actor", "critic"}
        missing_keys = required_keys - checkpoint.keys()

        if missing_keys:
            raise KeyError(
                f"Missing keys {missing_keys} in checkpoint: "
                f"{checkpoint_path}"
            )

        # Récupérer les paramètres de construction de l'agent
        args, kwargs = checkpoint["agent"]

        # Copier kwargs pour ne pas modifier les données chargées
        kwargs = dict(kwargs)

        # Désactiver le mode entraînement
        kwargs["training"] = False

        if algorithm == "MAPPO":
            # Réutiliser le critic partagé après le premier agent
            if shared_critic is not None:
                kwargs["critic"] = shared_critic

            agent = AgentClass(*args, **kwargs)

            if shared_critic is None:
                shared_critic = agent.critic

        else:
            # PPO et MADDPG utilisent leurs propres réseaux
            agent = AgentClass(*args, **kwargs)

        # Charger les poids de l'actor
        agent.actor.load_state_dict(
            checkpoint["actor"]
        )

        if algorithm == "MAPPO":
            # Charger le critic partagé une seule fois
            if agent_id == 0:
                agent.critic.load_state_dict(
                    checkpoint["critic"]
                )
        else:
            # Charger le critic individuel
            agent.critic.load_state_dict(
                checkpoint["critic"]
            )

        # Déplacer l'agent vers le périphérique demandé
        agent.to(device)

        # Activer le mode évaluation
        agent.eval()

        agents.append(agent)

    if algorithm == "MADDPG":
        # Chaque agent MADDPG doit connaître les autres agents
        for agent_id, agent in enumerate(agents):
            agent.agent_id = agent_id
            agent.set_agents(agents)

    # Associer chaque agent à son nom dans CybORG
    named_agents = {
        f"blue_agent_{agent_id}": agent
        for agent_id, agent in enumerate(agents)
    }

    return named_agents