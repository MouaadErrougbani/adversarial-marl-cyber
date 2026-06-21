# src/evaluation/eval.py

import json
import os
import time
from datetime import datetime
from statistics import mean, stdev

from joblib import Parallel, delayed
from tqdm import tqdm

from CybORG import CYBORG_VERSION

from src import make_env
from src.evaluation.builders import build_agents


class Submission:
    # Nom de la soumission
    NAME: str = "Evaluation"

    # Nom de l'équipe
    TEAM: str = "Mouaad"

    # Technique utilisée
    TECHNIQUE: str = (
        "Graph-based PPO With "
        "Intra-agent Communication"
    )


def build_global_observation(
    observations,
    num_agents,
):
    """
    Build the MAPPO global observation.
    """

    # Construire la liste ordonnée des agents
    agent_names = [
        f"blue_agent_{agent_id}"
        for agent_id in range(num_agents)
    ]

    # Vérifier que toutes les observations existent
    missing_agents = [
        agent_name
        for agent_name in agent_names
        if agent_name not in observations
    ]

    if missing_agents:
        raise KeyError(
            f"Missing MAPPO observations for agents: "
            f"{missing_agents}"
        )

    # Construire l'observation globale
    # comme pendant l'entraînement
    global_observation = [
        observations[agent_name][0]
        for agent_name in agent_names
    ]

    return global_observation


def extract_action(action_output):
    """
    Extract the action from the agent output.
    """

    # Certains agents retournent
    # (action, value, probability)
    if isinstance(action_output, tuple):
        return action_output[0]

    return action_output


def compute_episode_metrics(
    episode_index,
    step_rewards,
):
    """
    Compute the metrics of one episode.
    """

    # Retourner des valeurs nulles
    # lorsqu'aucune récompense n'existe
    if not step_rewards:
        return {
            "episode": episode_index + 1,
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
        }

    return {
        "episode": episode_index + 1,
        "avg_reward": float(
            mean(step_rewards)
        ),
        "std_reward": float(
            stdev(step_rewards)
            if len(step_rewards) > 1
            else 0.0
        ),
        "min_reward": float(
            min(step_rewards)
        ),
        "max_reward": float(
            max(step_rewards)
        ),
    }


def evaluate_one_episode(
    cyborg,
    wrapped_cyborg,
    agents,
    algorithm,
    episode_index,
    total_episodes,
    episode_length,
):
    """
    Evaluate one episode.
    """

    # Normaliser le nom de l'algorithme
    algorithm = algorithm.upper()

    # Réinitialiser l'environnement
    observations, _ = wrapped_cyborg.reset()

    # Stocker les récompenses et les actions
    step_rewards = []
    actions_log = []

    for step_index in tqdm(
        range(episode_length),
        desc=(
            f"Episode "
            f"{episode_index + 1}/{total_episodes}"
        ),
        leave=False,
    ):
        actions = {}

        if algorithm == "MAPPO":
            # Construire l'observation globale
            # comme pendant l'entraînement
            global_observation = (
                build_global_observation(
                    observations=observations,
                    num_agents=len(agents),
                )
            )

        for agent_name, agent in agents.items():
            # Ignorer les agents absents
            if (
                agent_name
                not in wrapped_cyborg.agents
                or agent_name
                not in observations
            ):
                continue

            local_observation, is_blocked = (
                observations[agent_name]
            )

            # Un agent bloqué continue
            # son action en cours
            if is_blocked:
                actions[agent_name] = None
                continue

            if algorithm == "MAPPO":
                # MAPPO reçoit les observations
                # locale et globale
                action_output = agent.get_action(
                    (
                        local_observation,
                        global_observation,
                        is_blocked,
                    )
                )
            else:
                # PPO et MADDPG reçoivent
                # l'observation locale
                action_output = agent.get_action(
                    (
                        local_observation,
                        is_blocked,
                    )
                )

            actions[agent_name] = (
                extract_action(action_output)
            )

        # Exécuter les actions
        (
            observations,
            rewards_by_agent,
            term,
            trunc,
            info,
        ) = wrapped_cyborg.step(actions)

        # Enregistrer les actions réellement exécutées
        actions_log.append(
            {
                agent_name: str(
                    cyborg.get_last_action(
                        agent_name
                    )
                )
                for agent_name
                in wrapped_cyborg.agents
            }
        )

        # Calculer la récompense moyenne
        # de l'équipe pour cette étape
        if rewards_by_agent:
            step_reward = mean(
                rewards_by_agent.values()
            )

            step_rewards.append(
                float(step_reward)
            )

        # Vérifier la fin de l'épisode
        done = {
            agent_name: (
                term.get(agent_name, False)
                or trunc.get(agent_name, False)
            )
            for agent_name
            in wrapped_cyborg.agents
        }

        if done and all(done.values()):
            break

    episode_metrics = compute_episode_metrics(
        episode_index=episode_index,
        step_rewards=step_rewards,
    )

    return episode_metrics, actions_log


def save_evaluation_results(
    log_path,
    algorithm,
    seed,
    max_eps,
    episode_length,
    workers,
    start,
    end,
    episode_metrics,
    actions_logs,
):
    """
    Save metrics and actions.
    """

    # Construire les chemins
    metrics_path = os.path.join(
        log_path,
        "metrics.json",
    )

    actions_path = os.path.join(
        log_path,
        "actions.json",
    )

    elapsed_time = end - start

    # Construire les métriques
    metrics_data = {
        "submission": {
            "author": Submission.NAME,
            "team": Submission.TEAM,
            "technique": Submission.TECHNIQUE,
        },
        "environment": {
            "cyborg_version": str(
                CYBORG_VERSION
            ),
            "scenario": "Scenario4",
        },
        "parameters": {
            "algorithm": algorithm,
            "seed": seed,
            "episode_length": episode_length,
            "max_episodes": max_eps,
            "workers": workers,
        },
        "time": {
            "start": str(start),
            "end": str(end),
            "elapsed": str(elapsed_time),
            "elapsed_seconds": (
                elapsed_time.total_seconds()
            ),
        },
        "episodes": episode_metrics,
    }

    # Construire le journal des actions
    actions_data = [
        {
            "episode": episode_index + 1,
            "actions": episode_actions,
        }
        for episode_index, episode_actions
        in enumerate(actions_logs)
    ]

    # Enregistrer les métriques
    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            metrics_data,
            output,
            indent=4,
        )

    # Enregistrer les actions
    with open(
        actions_path,
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            actions_data,
            output,
            indent=4,
        )

    print(
        f"Evaluation took {elapsed_time} "
        f"to finish"
    )

    print(
        f"Metrics saved to {metrics_path}"
    )

    print(
        f"Actions saved to {actions_path}"
    )


def run_evaluation(
    log_path,
    agents,
    algorithm,
    max_eps=100,
    seed=None,
    episode_length=500,
):
    """
    Run sequential evaluation.
    """

    # Construire un seul environnement
    wrapped_cyborg = make_env(
        seed=seed,
        steps=episode_length,
    )

    cyborg = wrapped_cyborg.env

    start = datetime.now()

    episode_metrics = []
    actions_logs = []

    # Exécuter les épisodes séquentiellement
    for episode_index in tqdm(
        range(max_eps),
        desc="Evaluation",
    ):
        metrics, actions = evaluate_one_episode(
            cyborg=cyborg,
            wrapped_cyborg=wrapped_cyborg,
            agents=agents,
            algorithm=algorithm,
            episode_index=episode_index,
            total_episodes=max_eps,
            episode_length=episode_length,
        )

        episode_metrics.append(metrics)
        actions_logs.append(actions)

    end = datetime.now()

    save_evaluation_results(
        log_path=log_path,
        algorithm=algorithm,
        seed=seed,
        max_eps=max_eps,
        episode_length=episode_length,
        workers=1,
        start=start,
        end=end,
        episode_metrics=episode_metrics,
        actions_logs=actions_logs,
    )

    return episode_metrics, actions_logs


def run_evaluation_parallel(
    log_path,
    agents,
    algorithm,
    max_eps=100,
    seed=None,
    workers=2,
    episode_length=500,
):
    """
    Run parallel evaluation.
    """

    # Construire les environnements parallèles
    envs = []

    for worker_id in range(workers):
        # Utiliser une seed différente
        # pour chaque environnement
        worker_seed = (
            None
            if seed is None
            else seed + worker_id
        )

        wrapped_cyborg = make_env(
            seed=worker_seed,
            steps=episode_length,
        )

        cyborg = wrapped_cyborg.env

        envs.append(
            (
                cyborg,
                wrapped_cyborg,
            )
        )

    start = datetime.now()

    # Exécuter les épisodes en parallèle
    outputs = Parallel(
        prefer="processes",
        n_jobs=workers,
    )(
        delayed(evaluate_one_episode)(
            *envs[
                episode_index % workers
            ],
            agents,
            algorithm,
            episode_index,
            max_eps,
            episode_length,
        )
        for episode_index in range(max_eps)
    )

    episode_metrics, actions_logs = zip(
        *outputs
    )

    episode_metrics = list(
        episode_metrics
    )

    actions_logs = list(
        actions_logs
    )

    end = datetime.now()

    save_evaluation_results(
        log_path=log_path,
        algorithm=algorithm,
        seed=seed,
        max_eps=max_eps,
        episode_length=episode_length,
        workers=workers,
        start=start,
        end=end,
        episode_metrics=episode_metrics,
        actions_logs=actions_logs,
    )

    return episode_metrics, actions_logs


def validate_eval_config(eval_cfg):
    """
    Validate the evaluation configuration.
    """

    required_keys = {
        "algorithm",
        "num_agents",
        "checkpoint_dir",
        "checkpoint_pattern",
        "device",
        "max_eps",
        "episode_length",
        "distribute",
        "append_timestamp",
        "output_path",
    }

    missing_keys = (
        required_keys - eval_cfg.keys()
    )

    if missing_keys:
        raise KeyError(
            f"Missing evaluation configuration keys: "
            f"{sorted(missing_keys)}"
        )

    if int(eval_cfg["max_eps"]) < 1:
        raise ValueError(
            "max_eps must be greater than zero"
        )

    if int(eval_cfg["episode_length"]) < 1:
        raise ValueError(
            "episode_length must be greater than zero"
        )

    if int(eval_cfg["distribute"]) < 1:
        raise ValueError(
            "distribute must be greater than zero"
        )


def run_eval(cfg):
    """
    Build agents and run the evaluation.
    """

    if (
        not isinstance(cfg, dict)
        or "eval" not in cfg
    ):
        raise ValueError(
            "The configuration must contain "
            "an 'eval' section"
        )

    eval_cfg = cfg["eval"]

    validate_eval_config(eval_cfg)

    algorithm = str(
        eval_cfg["algorithm"]
    ).upper()

    # Construire le chemin de sortie
    output_path = os.path.abspath(
        eval_cfg["output_path"]
    )

    # Ajouter un horodatage
    if eval_cfg["append_timestamp"]:
        output_path = os.path.join(
            output_path,
            time.strftime(
                "%Y%m%d_%H%M%S"
            ),
        )

    # Créer le dossier de sortie
    os.makedirs(
        output_path,
        exist_ok=True,
    )

    # Construire et charger les agents
    agents = build_agents(cfg)

    max_eps = int(
        eval_cfg["max_eps"]
    )

    seed = eval_cfg.get(
        "seed"
    )

    workers = int(
        eval_cfg["distribute"]
    )

    episode_length = int(
        eval_cfg["episode_length"]
    )

    if workers == 1:
        return run_evaluation(
            log_path=output_path,
            agents=agents,
            algorithm=algorithm,
            max_eps=max_eps,
            seed=seed,
            episode_length=episode_length,
        )

    return run_evaluation_parallel(
        log_path=output_path,
        agents=agents,
        algorithm=algorithm,
        max_eps=max_eps,
        seed=seed,
        workers=workers,
        episode_length=episode_length,
    )