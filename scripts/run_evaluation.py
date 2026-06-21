# scripts/run_evaluation.py

from copy import deepcopy
from pathlib import Path
import re

import yaml

from src.evaluation.eval import run_eval


CHECKPOINT_REGEX = re.compile(
    r"^(?P<group>[A-Za-z0-9_]+)-"
    r"(?P<agent_id>\d+)_checkpoint\.pt$"
)


def load_config(config_path):
    """
    Load a YAML configuration file.
    """

    # Convertir le chemin en objet Path
    config_path = Path(config_path)

    # Vérifier que le fichier existe
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    # Charger le fichier YAML
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        cfg = yaml.safe_load(
            config_file
        )

    # Vérifier la configuration
    if not isinstance(cfg, dict):
        raise ValueError(
            "The YAML configuration must "
            "contain a dictionary"
        )

    if (
        "eval" not in cfg
        or not isinstance(
            cfg["eval"],
            dict,
        )
    ):
        raise ValueError(
            "The configuration must contain "
            "an 'eval' section"
        )

    return cfg


def discover_checkpoint_groups(
    checkpoint_dir,
):
    """
    Discover checkpoint groups.
    """

    checkpoint_dir = Path(
        checkpoint_dir
    )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: "
            f"{checkpoint_dir}"
        )

    checkpoint_groups = {}

    for checkpoint_path in (
        checkpoint_dir.iterdir()
    ):
        if not checkpoint_path.is_file():
            continue

        match = CHECKPOINT_REGEX.match(
            checkpoint_path.name
        )

        if match is None:
            continue

        group_name = match.group(
            "group"
        )

        agent_id = int(
            match.group("agent_id")
        )

        checkpoint_groups.setdefault(
            group_name,
            set(),
        ).add(agent_id)

    return checkpoint_groups


def evaluate_group(
    base_cfg,
    group_name,
    base_output_path,
):
    """
    Evaluate one checkpoint group.
    """

    # Créer une copie indépendante
    group_cfg = deepcopy(
        base_cfg
    )

    algorithm = group_name.split(
        "_",
        maxsplit=1,
    )[0].upper()

    supported_algorithms = {
        "PPO",
        "MAPPO",
        "MADDPG",
    }

    if algorithm not in supported_algorithms:
        raise ValueError(
            f"Unsupported algorithm detected "
            f"from group '{group_name}': "
            f"{algorithm}"
        )

    # Configurer le groupe courant
    group_cfg["eval"]["algorithm"] = (
        algorithm
    )

    group_cfg["eval"][
        "checkpoint_pattern"
    ] = (
        f"{group_name}-"
        f"{{agent_id}}_checkpoint.pt"
    )

    group_cfg["eval"]["output_path"] = str(
        base_output_path / group_name
    )

    print("=" * 60)
    print(f"Evaluating {group_name}")
    print("=" * 60)

    episode_metrics, actions_logs = (
        run_eval(group_cfg)
    )

    print(
        f"Evaluated episodes for "
        f"{group_name}: "
        f"{len(episode_metrics)}"
    )

    return episode_metrics, actions_logs


def main(
    path="configs/eval.yaml",
):
    """
    Run evaluation from a YAML configuration.
    """

    cfg = load_config(path)

    eval_cfg = cfg["eval"]

    algorithm = str(
        eval_cfg.get(
            "algorithm",
            "",
        )
    ).strip().upper()

    if algorithm != "AUTO":
        episode_metrics, actions_logs = (
            run_eval(cfg)
        )

        print(
            f"Evaluated episodes: "
            f"{len(episode_metrics)}"
        )

        return {
            algorithm: {
                "metrics": episode_metrics,
                "actions": actions_logs,
            }
        }

    checkpoint_dir = Path(
        eval_cfg["checkpoint_dir"]
    )

    base_output_path = Path(
        eval_cfg["output_path"]
    )

    checkpoint_groups = (
        discover_checkpoint_groups(
            checkpoint_dir
        )
    )

    if not checkpoint_groups:
        raise FileNotFoundError(
            f"No valid checkpoints found in: "
            f"{checkpoint_dir}"
        )

    num_agents = int(
        eval_cfg["num_agents"]
    )

    expected_agent_ids = set(
        range(num_agents)
    )

    all_results = {}

    for group_name in sorted(
        checkpoint_groups
    ):
        available_agent_ids = (
            checkpoint_groups[group_name]
        )

        missing_agent_ids = (
            expected_agent_ids
            - available_agent_ids
        )

        if missing_agent_ids:
            print(
                f"Skipping {group_name}: "
                f"missing checkpoints for agents "
                f"{sorted(missing_agent_ids)}"
            )
            continue

        try:
            episode_metrics, actions_logs = (
                evaluate_group(
                    base_cfg=cfg,
                    group_name=group_name,
                    base_output_path=(
                        base_output_path
                    ),
                )
            )
        except Exception as error:
            print(
                f"Evaluation failed for "
                f"{group_name}: {error}"
            )
            continue

        all_results[group_name] = {
            "metrics": episode_metrics,
            "actions": actions_logs,
        }

    if not all_results:
        raise RuntimeError(
            "No checkpoint group was "
            "successfully evaluated"
        )

    return all_results


if __name__ == "__main__":
    import argparse

    # Créer le parseur
    parser = argparse.ArgumentParser(
        description=(
            "CybORG Evaluation Script"
        )
    )

    # Définir le fichier de configuration
    parser.add_argument(
        "--path",
        type=str,
        default="configs/eval.yaml",
        help=(
            "Path to the YAML "
            "configuration file"
        ),
    )

    args = parser.parse_args()

    main(
        path=args.path
    )