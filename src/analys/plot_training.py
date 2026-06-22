
"""
plot_training_results.py

Génération des figures utiles pour le chapitre 6 — Résultats et discussion.

Hypothèse importante
--------------------
L'environnement retourne une récompense partagée (shared reward) commune aux cinq
agents Blue. Par conséquent :

- avg_reward est la métrique de performance de l'équipe ;
- avg_reward_agent_i ne doit PAS être interprétée comme une récompense individuelle ;
- les différences entre agents sont étudiées uniquement à travers les pertes enregistrées :
  actor_loss_agent_i, critic_loss_agent_i et total_loss_agent_i.

Fichiers attendus
-----------------
logs/
├── mappo_gat_gat.pt
├── mappo_gcn_gcn.pt
├── ppo_gat_gat.pt
└── ppo_gcn_gcn.pt

Sorties principales
-------------------
plot_results/
├── rewards/
├── losses/
├── timing/
└── tables/

Le script produit notamment :
1. les courbes de récompense partagée de l'équipe ;
2. les comparaisons MAPPO/IPPO à encodeur fixé ;
3. les comparaisons GAT/GCN à algorithme fixé ;
4. les détails moyenne, écart-type, minimum et maximum par configuration ;
5. les performances sur la fin de l'entraînement ;
6. les pertes Actor, Critic et totales de chaque agent ;
7. les temps de collecte et d'optimisation ;
8. des tableaux CSV directement exploitables dans le mémoire.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Configuration générale
# ============================================================

EXPECTED_CONFIGURATIONS = {
    "mappo_gat_gat": "MAPPO-GAT",
    "mappo_gcn_gcn": "MAPPO-GCN",
    "ppo_gat_gat": "IPPO-GAT",
    "ppo_gcn_gcn": "IPPO-GCN",
}

NUM_AGENTS = 5


# ============================================================
# Chargement et validation
# ============================================================

def load_logs(log_directory: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Charge tous les fichiers .pt valides du dossier de logs."""

    log_directory = Path(log_directory)

    if not log_directory.exists():
        raise FileNotFoundError(
            f"Le dossier de logs n'existe pas : {log_directory}"
        )

    logs: dict[str, list[dict[str, Any]]] = {}

    for file_path in sorted(log_directory.glob("*.pt")):
        history = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(history, list):
            print(
                f"[WARNING] {file_path.name} ignoré : "
                "le contenu n'est pas une liste."
            )
            continue

        logs[file_path.stem] = history
        print(
            f"[INFO] {file_path.stem}: "
            f"{len(history)} itérations chargées"
        )

    if not logs:
        raise ValueError(
            f"Aucun fichier .pt valide trouvé dans {log_directory}"
        )

    return logs


def display_name(model_name: str) -> str:
    """Transforme le nom interne en nom lisible."""

    lower_name = model_name.lower()

    algorithm = (
        "MAPPO"
        if lower_name.startswith("mappo")
        else "IPPO"
    )

    if "gat" in lower_name:
        encoder = "GAT"
    elif "gcn" in lower_name:
        encoder = "GCN"
    else:
        encoder = "Unknown"

    return f"{algorithm}-{encoder}"


def validate_shared_rewards(
    logs: dict[str, list[dict[str, Any]]],
    num_agents: int = NUM_AGENTS,
    tolerance: float = 1e-6,
) -> None:
    """
    Vérifie que avg_reward_agent_i correspond bien à avg_reward.

    Cette validation évite d'utiliser par erreur les récompenses partagées
    comme des récompenses individuelles.
    """

    print("\n" + "=" * 78)
    print("VALIDATION DE LA RÉCOMPENSE PARTAGÉE")
    print("=" * 78)

    for model_name, history in logs.items():
        checked = 0
        mismatches = 0

        for step in history:
            team_reward = safe_float(step.get("avg_reward"))

            if team_reward is None:
                continue

            for agent_index in range(num_agents):
                agent_reward = safe_float(
                    step.get(f"avg_reward_agent_{agent_index}")
                )

                if agent_reward is None:
                    continue

                checked += 1

                if abs(agent_reward - team_reward) > tolerance:
                    mismatches += 1

        if checked == 0:
            print(
                f"[WARNING] {display_name(model_name)} : "
                "aucune valeur avg_reward_agent_i vérifiable."
            )
        elif mismatches == 0:
            print(
                f"[OK] {display_name(model_name)} : "
                "les récompenses par agent sont identiques à la récompense "
                "partagée. Elles ne seront pas tracées séparément."
            )
        else:
            print(
                f"[WARNING] {display_name(model_name)} : "
                f"{mismatches}/{checked} valeurs diffèrent de avg_reward. "
                "Vérifiez la logique de journalisation."
            )


# ============================================================
# Utilitaires numériques
# ============================================================

def safe_float(value: Any) -> float | None:
    """Convertit une valeur en float fini, sinon retourne None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def extract_metric(
    history: list[dict[str, Any]],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait une métrique en conservant les vrais indices d'itération.
    """

    iterations: list[int] = []
    values: list[float] = []

    for index, step in enumerate(history):
        value = safe_float(step.get(metric_name))

        if value is None:
            continue

        iterations.append(index)
        values.append(value)

    return (
        np.asarray(iterations, dtype=int),
        np.asarray(values, dtype=float),
    )


def moving_average(
    values: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule une moyenne mobile simple."""

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    if window_size <= 1 or values.size < window_size:
        return np.arange(values.size), values.copy()

    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(values, kernel, mode="valid")
    indices = np.arange(window_size - 1, values.size)

    return indices, smoothed


def align_metrics(
    history: list[dict[str, Any]],
    metric_names: Iterable[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Aligne plusieurs métriques sur les itérations où elles sont toutes valides.
    """

    names = list(metric_names)
    iterations: list[int] = []
    collected: dict[str, list[float]] = {
        name: [] for name in names
    }

    for index, step in enumerate(history):
        row: dict[str, float] = {}
        valid = True

        for name in names:
            value = safe_float(step.get(name))

            if value is None:
                valid = False
                break

            row[name] = value

        if not valid:
            continue

        iterations.append(index)

        for name in names:
            collected[name].append(row[name])

    return (
        np.asarray(iterations, dtype=int),
        {
            name: np.asarray(values, dtype=float)
            for name, values in collected.items()
        },
    )


def get_selected_logs(
    logs: dict[str, list[dict[str, Any]]],
    algorithm: str | None = None,
    architecture: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Sélectionne les configurations demandées."""

    selected: dict[str, list[dict[str, Any]]] = {}

    for model_name, history in logs.items():
        lower_name = model_name.lower()

        if algorithm == "mappo":
            if not lower_name.startswith("mappo"):
                continue

        if algorithm == "ippo":
            if not (
                lower_name.startswith("ppo")
                or lower_name.startswith("ippo")
            ):
                continue

        if architecture is not None:
            if architecture.lower() not in lower_name:
                continue

        selected[model_name] = history

    return selected


def ensure_parent_directory(path: str | Path) -> Path:
    """Crée le dossier parent et retourne le chemin."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    figure: plt.Figure,
    save_path: str | Path,
) -> None:
    """Enregistre et ferme une figure."""

    save_path = ensure_parent_directory(save_path)

    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {save_path}")


# ============================================================
# Récompense partagée de l'équipe
# ============================================================

def plot_all_team_rewards(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 10,
) -> None:
    """
    Compare les quatre configurations sur une seule figure.

    Les lignes fines sont les valeurs brutes.
    Les lignes épaisses sont les moyennes mobiles.
    """

    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = 0

    for model_name, history in logs.items():
        iterations, rewards = extract_metric(
            history,
            "avg_reward",
        )

        if rewards.size == 0:
            continue

        raw_line = axis.plot(
            iterations,
            rewards,
            linewidth=1,
            alpha=0.25,
        )[0]

        smooth_positions, smooth_rewards = moving_average(
            rewards,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_rewards,
            linewidth=2.5,
            label=display_name(model_name),
        )

        plotted += 1

    if plotted:
        axis.legend()

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        "Évolution de la récompense partagée pendant l'entraînement"
    )
    axis.set_xlabel("Itération d'entraînement")
    axis.set_ylabel("Récompense moyenne de l'équipe")
    axis.grid(True, linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


def plot_reward_details_per_configuration(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 10,
) -> None:
    """
    Crée une figure séparée pour chaque configuration avec :
    moyenne, moyenne ± écart-type, minimum, maximum et moyenne mobile.
    """

    output_directory = Path(output_directory)

    for model_name, history in logs.items():
        iterations, metrics = align_metrics(
            history,
            [
                "avg_reward",
                "std_reward",
                "min_reward",
                "max_reward",
            ],
        )

        if iterations.size == 0:
            continue

        avg_rewards = metrics["avg_reward"]
        std_rewards = metrics["std_reward"]
        min_rewards = metrics["min_reward"]
        max_rewards = metrics["max_reward"]

        figure, axis = plt.subplots(figsize=(12, 7))

        axis.plot(
            iterations,
            avg_rewards,
            linewidth=1.2,
            alpha=0.7,
            label="Récompense moyenne",
        )

        axis.fill_between(
            iterations,
            avg_rewards - std_rewards,
            avg_rewards + std_rewards,
            alpha=0.2,
            label="Moyenne ± 1 écart-type",
        )

        axis.plot(
            iterations,
            min_rewards,
            linestyle="--",
            linewidth=1.1,
            label="Récompense minimale",
        )

        axis.plot(
            iterations,
            max_rewards,
            linestyle="--",
            linewidth=1.1,
            label="Récompense maximale",
        )

        smooth_positions, smooth_rewards = moving_average(
            avg_rewards,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_rewards,
            linewidth=2.6,
            label=f"Moyenne mobile ({window_size})",
        )

        axis.axhline(y=0, linestyle=":", linewidth=1)
        axis.set_title(
            f"Récompense partagée — {display_name(model_name)}"
        )
        axis.set_xlabel("Itération d'entraînement")
        axis.set_ylabel("Récompense de l'équipe")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()

        save_figure(
            figure,
            output_directory
            / f"reward_details_{model_name}.png",
        )


def plot_reward_comparison(
    selected_logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    title: str,
    window_size: int = 10,
) -> None:
    """Compare deux configurations compatibles."""

    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = 0

    for model_name, history in selected_logs.items():
        iterations, rewards = extract_metric(
            history,
            "avg_reward",
        )

        if rewards.size == 0:
            continue

        raw_line = axis.plot(
            iterations,
            rewards,
            linewidth=1,
            alpha=0.25,
        )[0]

        smooth_positions, smooth_rewards = moving_average(
            rewards,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_rewards,
            linewidth=2.7,
            label=display_name(model_name),
        )

        plotted += 1

    if plotted:
        axis.legend()

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Itération d'entraînement")
    axis.set_ylabel("Récompense moyenne partagée")
    axis.grid(True, linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


def plot_pairwise_reward_comparisons(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 10,
) -> None:
    """Produit les quatre comparaisons contrôlées du mémoire."""

    output_directory = Path(output_directory)

    plot_reward_comparison(
        get_selected_logs(logs, architecture="gat"),
        output_directory / "mappo_vs_ippo_gat.png",
        "Comparaison MAPPO–IPPO avec encodeur GAT",
        window_size,
    )

    plot_reward_comparison(
        get_selected_logs(logs, architecture="gcn"),
        output_directory / "mappo_vs_ippo_gcn.png",
        "Comparaison MAPPO–IPPO avec encodeur GCN",
        window_size,
    )

    plot_reward_comparison(
        get_selected_logs(logs, algorithm="mappo"),
        output_directory / "gat_vs_gcn_mappo.png",
        "Comparaison GAT–GCN avec MAPPO",
        window_size,
    )

    plot_reward_comparison(
        get_selected_logs(logs, algorithm="ippo"),
        output_directory / "gat_vs_gcn_ippo.png",
        "Comparaison GAT–GCN avec IPPO",
        window_size,
    )


def plot_final_training_performance(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    last_n_iterations: int = 50,
) -> None:
    """
    Compare la moyenne de avg_reward sur les dernières itérations.

    Attention : les barres d'erreur représentent la dispersion entre les
    valeurs moyennes enregistrées à la fin de l'entraînement. Elles ne
    représentent pas une dispersion entre plusieurs graines indépendantes.
    """

    names: list[str] = []
    means: list[float] = []
    standard_deviations: list[float] = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(history, "avg_reward")

        if rewards.size == 0:
            continue

        count = min(last_n_iterations, rewards.size)
        final_rewards = rewards[-count:]

        names.append(display_name(model_name))
        means.append(float(np.mean(final_rewards)))
        standard_deviations.append(float(np.std(final_rewards)))

    if not names:
        return

    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(11, 7))

    bars = axis.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6,
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        "Performance en fin d'entraînement\n"
        f"Moyenne des {last_n_iterations} dernières itérations"
    )
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Récompense moyenne partagée")
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, mean_value in zip(bars, means):
        axis.annotate(
            f"{mean_value:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5 if mean_value >= 0 else -16),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean_value >= 0 else "top",
        )

    save_figure(figure, save_path)


def plot_final_reward_boxplot(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    last_n_iterations: int = 50,
) -> None:
    """
    Boxplot des avg_reward sur les dernières itérations.

    Ce graphique décrit la stabilité temporelle de la fin d'entraînement.
    Il ne remplace pas une analyse sur plusieurs graines indépendantes.
    """

    names: list[str] = []
    samples: list[np.ndarray] = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(history, "avg_reward")

        if rewards.size == 0:
            continue

        count = min(last_n_iterations, rewards.size)
        names.append(display_name(model_name))
        samples.append(rewards[-count:])

    if not samples:
        return

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.boxplot(samples, labels=names, showmeans=True)
    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        "Distribution des récompenses en fin d'entraînement\n"
        f"{last_n_iterations} dernières itérations"
    )
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Récompense moyenne partagée")
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


# ============================================================
# Pertes par agent
# ============================================================

def plot_loss_per_agent(
    history: list[dict[str, Any]],
    model_name: str,
    loss_prefix: str,
    save_path: str | Path,
    title_label: str,
    window_size: int = 10,
    num_agents: int = NUM_AGENTS,
) -> None:
    """
    Trace une perte donnée pour les cinq agents d'une configuration.
    """

    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = 0

    for agent_index in range(num_agents):
        metric_name = f"{loss_prefix}_agent_{agent_index}"

        iterations, values = extract_metric(
            history,
            metric_name,
        )

        if values.size == 0:
            continue

        smooth_positions, smooth_values = moving_average(
            values,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_values,
            linewidth=2,
            label=f"Agent B{agent_index}",
        )

        plotted += 1

    if plotted:
        axis.legend(ncol=2)

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        f"{title_label} par agent — {display_name(model_name)}"
    )
    axis.set_xlabel("Itération d'entraînement")
    axis.set_ylabel(title_label)
    axis.grid(True, linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


def plot_all_agent_losses(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 10,
) -> None:
    """
    Produit trois figures par configuration :
    Actor loss, Critic loss et Total loss par agent.
    """

    output_directory = Path(output_directory)

    for model_name, history in logs.items():
        plot_loss_per_agent(
            history,
            model_name,
            "actor_loss",
            output_directory
            / f"actor_loss_agents_{model_name}.png",
            "Actor loss",
            window_size,
        )

        plot_loss_per_agent(
            history,
            model_name,
            "critic_loss",
            output_directory
            / f"critic_loss_agents_{model_name}.png",
            "Critic loss",
            window_size,
        )

        plot_loss_per_agent(
            history,
            model_name,
            "total_loss",
            output_directory
            / f"total_loss_agents_{model_name}.png",
            "Total loss",
            window_size,
        )


def plot_average_loss_comparison(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 10,
) -> None:
    """
    Compare avg_loss entre les configurations.

    Cette figure sert uniquement à décrire la dynamique d'optimisation.
    Elle ne doit pas être utilisée seule pour classer les performances.
    """

    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = 0

    for model_name, history in logs.items():
        iterations, values = extract_metric(
            history,
            "avg_loss",
        )

        if values.size == 0:
            continue

        smooth_positions, smooth_values = moving_average(
            values,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_values,
            linewidth=2.4,
            label=display_name(model_name),
        )

        plotted += 1

    if plotted:
        axis.legend()

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        "Évolution de la perte moyenne pendant l'entraînement"
    )
    axis.set_xlabel("Itération d'entraînement")
    axis.set_ylabel("Perte moyenne")
    axis.grid(True, linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


def plot_final_agent_loss_bars(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    last_n_iterations: int = 50,
    num_agents: int = NUM_AGENTS,
) -> None:
    """
    Produit un diagramme en barres par configuration et par type de perte.

    Chaque barre est la moyenne de la perte de l'agent sur les dernières
    itérations.
    """

    output_directory = Path(output_directory)

    loss_definitions = [
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("total_loss", "Total loss"),
    ]

    for model_name, history in logs.items():
        for loss_prefix, title_label in loss_definitions:
            agent_names: list[str] = []
            means: list[float] = []

            for agent_index in range(num_agents):
                _, values = extract_metric(
                    history,
                    f"{loss_prefix}_agent_{agent_index}",
                )

                if values.size == 0:
                    continue

                count = min(last_n_iterations, values.size)

                agent_names.append(f"B{agent_index}")
                means.append(
                    float(np.mean(values[-count:]))
                )

            if not agent_names:
                continue

            positions = np.arange(len(agent_names))

            figure, axis = plt.subplots(figsize=(10, 6))
            bars = axis.bar(positions, means)

            axis.set_xticks(positions)
            axis.set_xticklabels(agent_names)
            axis.axhline(y=0, linestyle=":", linewidth=1)
            axis.set_title(
                f"{title_label} moyenne en fin d'entraînement\n"
                f"{display_name(model_name)} — "
                f"{last_n_iterations} dernières itérations"
            )
            axis.set_xlabel("Agent")
            axis.set_ylabel(title_label)
            axis.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.5,
            )

            for bar, value in zip(bars, means):
                axis.annotate(
                    f"{value:.3f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    xytext=(0, 5 if value >= 0 else -16),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                )

            save_figure(
                figure,
                output_directory
                / f"final_{loss_prefix}_{model_name}.png",
            )


# ============================================================
# Temps de calcul
# ============================================================

def plot_reward_vs_cumulative_time(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 10,
) -> None:
    """
    Compare la récompense selon le temps cumulé collecte + optimisation.
    """

    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = 0

    for model_name, history in logs.items():
        rewards: list[float] = []
        times: list[float] = []

        for step in history:
            reward = safe_float(step.get("avg_reward"))
            collection_time = safe_float(
                step.get("collection_time_sec")
            )
            training_time = safe_float(
                step.get("training_time_sec")
            )

            if reward is None:
                continue

            collection_time = collection_time or 0.0
            training_time = training_time or 0.0

            total_time = max(
                0.0,
                collection_time + training_time,
            )

            rewards.append(reward)
            times.append(total_time)

        if not rewards:
            continue

        rewards_array = np.asarray(rewards, dtype=float)
        cumulative_minutes = (
            np.cumsum(np.asarray(times, dtype=float)) / 60.0
        )

        smooth_positions, smooth_rewards = moving_average(
            rewards_array,
            window_size=window_size,
        )

        axis.plot(
            cumulative_minutes[smooth_positions],
            smooth_rewards,
            linewidth=2.5,
            label=display_name(model_name),
        )

        plotted += 1

    if plotted:
        axis.legend()

    axis.set_title(
        "Efficacité temporelle : récompense et temps cumulé"
    )
    axis.set_xlabel("Temps cumulé (minutes)")
    axis.set_ylabel("Récompense moyenne partagée")
    axis.grid(True, linestyle="--", alpha=0.5)

    save_figure(figure, save_path)


def plot_total_time_comparison(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
) -> None:
    """Compare les temps totaux de collecte et d'optimisation."""

    names: list[str] = []
    collection_minutes: list[float] = []
    training_minutes: list[float] = []

    for model_name, history in logs.items():
        collection_total = 0.0
        training_total = 0.0

        for step in history:
            collection_total += (
                safe_float(step.get("collection_time_sec"))
                or 0.0
            )
            training_total += (
                safe_float(step.get("training_time_sec"))
                or 0.0
            )

        names.append(display_name(model_name))
        collection_minutes.append(collection_total / 60.0)
        training_minutes.append(training_total / 60.0)

    if not names:
        return

    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(11, 7))

    axis.bar(
        positions,
        collection_minutes,
        label="Collecte",
    )

    axis.bar(
        positions,
        training_minutes,
        bottom=collection_minutes,
        label="Optimisation",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.set_title("Temps total par configuration")
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Temps total (minutes)")
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)
    axis.legend()

    save_figure(figure, save_path)


# ============================================================
# Tableaux CSV pour le mémoire
# ============================================================

def export_training_summary_csv(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    last_n_iterations: int = 50,
) -> None:
    """Exporte un résumé global des quatre configurations."""

    save_path = ensure_parent_directory(save_path)

    fieldnames = [
        "configuration",
        "num_iterations",
        "global_reward_mean",
        "global_reward_std",
        "final_reward_mean",
        "final_reward_std",
        "best_recorded_avg_reward",
        "worst_recorded_avg_reward",
        "total_collection_minutes",
        "total_training_minutes",
        "total_minutes",
    ]

    rows: list[dict[str, Any]] = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(history, "avg_reward")

        if rewards.size == 0:
            continue

        count = min(last_n_iterations, rewards.size)
        final_rewards = rewards[-count:]

        collection_seconds = 0.0
        training_seconds = 0.0

        for step in history:
            collection_seconds += (
                safe_float(step.get("collection_time_sec"))
                or 0.0
            )
            training_seconds += (
                safe_float(step.get("training_time_sec"))
                or 0.0
            )

        rows.append(
            {
                "configuration": display_name(model_name),
                "num_iterations": int(rewards.size),
                "global_reward_mean": float(np.mean(rewards)),
                "global_reward_std": float(np.std(rewards)),
                "final_reward_mean": float(np.mean(final_rewards)),
                "final_reward_std": float(np.std(final_rewards)),
                "best_recorded_avg_reward": float(np.max(rewards)),
                "worst_recorded_avg_reward": float(np.min(rewards)),
                "total_collection_minutes": collection_seconds / 60.0,
                "total_training_minutes": training_seconds / 60.0,
                "total_minutes": (
                    collection_seconds + training_seconds
                ) / 60.0,
            }
        )

    rows.sort(
        key=lambda row: row["final_reward_mean"],
        reverse=True,
    )

    with save_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Tableau enregistré : {save_path}")


def export_agent_loss_summary_csv(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    last_n_iterations: int = 50,
    num_agents: int = NUM_AGENTS,
) -> None:
    """Exporte les pertes finales moyennes de chaque agent."""

    save_path = ensure_parent_directory(save_path)

    fieldnames = [
        "configuration",
        "agent",
        "actor_loss_final_mean",
        "actor_loss_final_std",
        "critic_loss_final_mean",
        "critic_loss_final_std",
        "total_loss_final_mean",
        "total_loss_final_std",
    ]

    rows: list[dict[str, Any]] = []

    for model_name, history in logs.items():
        for agent_index in range(num_agents):
            row: dict[str, Any] = {
                "configuration": display_name(model_name),
                "agent": f"B{agent_index}",
            }

            available = False

            for prefix in [
                "actor_loss",
                "critic_loss",
                "total_loss",
            ]:
                _, values = extract_metric(
                    history,
                    f"{prefix}_agent_{agent_index}",
                )

                if values.size == 0:
                    row[f"{prefix}_final_mean"] = ""
                    row[f"{prefix}_final_std"] = ""
                    continue

                available = True
                count = min(last_n_iterations, values.size)
                final_values = values[-count:]

                row[f"{prefix}_final_mean"] = float(
                    np.mean(final_values)
                )
                row[f"{prefix}_final_std"] = float(
                    np.std(final_values)
                )

            if available:
                rows.append(row)

    with save_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Tableau enregistré : {save_path}")


# ============================================================
# Résumé terminal
# ============================================================

def print_training_summary(
    logs: dict[str, list[dict[str, Any]]],
    last_n_iterations: int = 50,
) -> None:
    """Affiche le classement final selon la récompense partagée."""

    print("\n" + "=" * 78)
    print("RÉSUMÉ DES PERFORMANCES D'ENTRAÎNEMENT")
    print("=" * 78)

    results: list[dict[str, Any]] = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(history, "avg_reward")

        if rewards.size == 0:
            continue

        count = min(last_n_iterations, rewards.size)
        final_rewards = rewards[-count:]

        total_seconds = 0.0

        for step in history:
            total_seconds += (
                safe_float(step.get("collection_time_sec"))
                or 0.0
            )
            total_seconds += (
                safe_float(step.get("training_time_sec"))
                or 0.0
            )

        results.append(
            {
                "name": display_name(model_name),
                "iterations": int(rewards.size),
                "final_mean": float(np.mean(final_rewards)),
                "final_std": float(np.std(final_rewards)),
                "best": float(np.max(rewards)),
                "worst": float(np.min(rewards)),
                "total_minutes": total_seconds / 60.0,
            }
        )

    results.sort(
        key=lambda item: item["final_mean"],
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['name']}")
        print(
            f"   Itérations                  : "
            f"{result['iterations']}"
        )
        print(
            f"   Moyenne finale partagée     : "
            f"{result['final_mean']:.4f}"
        )
        print(
            f"   Écart-type final temporel   : "
            f"{result['final_std']:.4f}"
        )
        print(
            f"   Meilleure moyenne observée  : "
            f"{result['best']:.4f}"
        )
        print(
            f"   Pire moyenne observée       : "
            f"{result['worst']:.4f}"
        )
        print(
            f"   Temps cumulé total (min)    : "
            f"{result['total_minutes']:.2f}"
        )

    print("\n" + "=" * 78)



# ============================================================
# Figures regroupées pour le mémoire
# ============================================================

def plot_grouped_reward_details(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 10,
) -> None:
    """
    Regroupe les détails de récompense des quatre configurations
    dans une seule figure 2 x 2.
    """

    ordered_items = sorted(
        logs.items(),
        key=lambda item: display_name(item[0]),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for axis, (model_name, history) in zip(
        axes_flat,
        ordered_items,
    ):
        iterations, metrics = align_metrics(
            history,
            [
                "avg_reward",
                "std_reward",
                "min_reward",
                "max_reward",
            ],
        )

        if iterations.size == 0:
            axis.text(
                0.5,
                0.5,
                "Aucune donnée disponible",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(display_name(model_name))
            continue

        avg_rewards = metrics["avg_reward"]
        std_rewards = metrics["std_reward"]
        min_rewards = metrics["min_reward"]
        max_rewards = metrics["max_reward"]

        axis.plot(
            iterations,
            avg_rewards,
            linewidth=1.1,
            alpha=0.65,
            label="Moyenne",
        )
        axis.fill_between(
            iterations,
            avg_rewards - std_rewards,
            avg_rewards + std_rewards,
            alpha=0.18,
            label="Moyenne ± 1 écart-type",
        )
        axis.plot(
            iterations,
            min_rewards,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="Minimum",
        )
        axis.plot(
            iterations,
            max_rewards,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="Maximum",
        )

        smooth_positions, smooth_rewards = moving_average(
            avg_rewards,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_rewards,
            linewidth=2.5,
            label=f"Moyenne mobile ({window_size})",
        )

        axis.axhline(y=0, linestyle=":", linewidth=1)
        axis.set_title(display_name(model_name))
        axis.set_xlabel("Itération")
        axis.set_ylabel("Récompense partagée")
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(fontsize=8)

    for axis in axes_flat[len(ordered_items):]:
        axis.set_visible(False)

    figure.suptitle(
        "Détails de la récompense partagée par configuration",
        fontsize=17,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = ensure_parent_directory(save_path)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[INFO] Figure enregistrée : {save_path}")


def _draw_reward_comparison_on_axis(
    axis: plt.Axes,
    selected_logs: dict[str, list[dict[str, Any]]],
    title: str,
    window_size: int,
) -> None:
    """Dessine une comparaison de récompense sur un axe existant."""

    plotted = 0

    for model_name, history in selected_logs.items():
        iterations, rewards = extract_metric(
            history,
            "avg_reward",
        )

        if rewards.size == 0:
            continue

        raw_line = axis.plot(
            iterations,
            rewards,
            linewidth=1,
            alpha=0.22,
        )[0]

        smooth_positions, smooth_rewards = moving_average(
            rewards,
            window_size=window_size,
        )

        axis.plot(
            iterations[smooth_positions],
            smooth_rewards,
            linewidth=2.5,
            color=raw_line.get_color(),
            label=display_name(model_name),
        )
        plotted += 1

    if plotted:
        axis.legend(fontsize=9)

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Itération")
    axis.set_ylabel("Récompense moyenne partagée")
    axis.grid(True, linestyle="--", alpha=0.45)


def plot_grouped_pairwise_reward_comparisons(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 10,
) -> None:
    """
    Regroupe les quatre comparaisons contrôlées dans une figure 2 x 2.
    """

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12),
        squeeze=False,
    )

    _draw_reward_comparison_on_axis(
        axes[0, 0],
        get_selected_logs(logs, architecture="gat"),
        "MAPPO vs IPPO — encodeur GAT",
        window_size,
    )

    _draw_reward_comparison_on_axis(
        axes[0, 1],
        get_selected_logs(logs, architecture="gcn"),
        "MAPPO vs IPPO — encodeur GCN",
        window_size,
    )

    _draw_reward_comparison_on_axis(
        axes[1, 0],
        get_selected_logs(logs, algorithm="mappo"),
        "GAT vs GCN — MAPPO",
        window_size,
    )

    _draw_reward_comparison_on_axis(
        axes[1, 1],
        get_selected_logs(logs, algorithm="ippo"),
        "GAT vs GCN — IPPO",
        window_size,
    )

    figure.suptitle(
        "Comparaisons contrôlées des récompenses d'entraînement",
        fontsize=17,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = ensure_parent_directory(save_path)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[INFO] Figure enregistrée : {save_path}")


def plot_grouped_loss_type(
    logs: dict[str, list[dict[str, Any]]],
    loss_prefix: str,
    title_label: str,
    save_path: str | Path,
    window_size: int = 10,
    num_agents: int = NUM_AGENTS,
) -> None:
    """
    Regroupe un même type de perte dans une figure 2 x 2.

    Chaque sous-figure correspond à une configuration.
    Les cinq courbes correspondent aux agents B0 à B4.
    """

    ordered_items = sorted(
        logs.items(),
        key=lambda item: display_name(item[0]),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, (model_name, history) in zip(
        axes_flat,
        ordered_items,
    ):
        plotted = 0

        for agent_index in range(num_agents):
            iterations, values = extract_metric(
                history,
                f"{loss_prefix}_agent_{agent_index}",
            )

            if values.size == 0:
                continue

            smooth_positions, smooth_values = moving_average(
                values,
                window_size=window_size,
            )

            axis.plot(
                iterations[smooth_positions],
                smooth_values,
                linewidth=1.8,
                label=f"B{agent_index}",
            )
            plotted += 1

        if plotted:
            axis.legend(
                ncol=3,
                fontsize=8,
            )

        axis.axhline(y=0, linestyle=":", linewidth=1)
        axis.set_title(display_name(model_name))
        axis.set_xlabel("Itération")
        axis.set_ylabel(title_label)
        axis.grid(True, linestyle="--", alpha=0.45)

    for axis in axes_flat[len(ordered_items):]:
        axis.set_visible(False)

    figure.suptitle(
        f"{title_label} par agent et par configuration",
        fontsize=17,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = ensure_parent_directory(save_path)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[INFO] Figure enregistrée : {save_path}")


def plot_grouped_agent_losses(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 10,
) -> None:
    """
    Produit trois figures regroupées :
    - Actor loss : 4 sous-figures ;
    - Critic loss : 4 sous-figures ;
    - Total loss : 4 sous-figures.
    """

    output_directory = Path(output_directory)

    plot_grouped_loss_type(
        logs,
        loss_prefix="actor_loss",
        title_label="Actor loss",
        save_path=(
            output_directory
            / "grouped_actor_loss_agents.png"
        ),
        window_size=window_size,
    )

    plot_grouped_loss_type(
        logs,
        loss_prefix="critic_loss",
        title_label="Critic loss",
        save_path=(
            output_directory
            / "grouped_critic_loss_agents.png"
        ),
        window_size=window_size,
    )

    plot_grouped_loss_type(
        logs,
        loss_prefix="total_loss",
        title_label="Total loss",
        save_path=(
            output_directory
            / "grouped_total_loss_agents.png"
        ),
        window_size=window_size,
    )


# ============================================================
# Génération complète
# ============================================================

def generate_result_figures(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path = "plot_results",
    reward_window_size: int = 10,
    loss_window_size: int = 10,
    last_n_iterations: int = 50,
) -> None:
    """Génère toutes les figures et tous les tableaux utiles."""

    output_directory = Path(output_directory)

    rewards_directory = output_directory / "rewards"
    losses_directory = output_directory / "losses"
    timing_directory = output_directory / "timing"
    tables_directory = output_directory / "tables"

    for directory in [
        rewards_directory,
        losses_directory,
        timing_directory,
        tables_directory,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    validate_shared_rewards(logs)
    print_training_summary(
        logs,
        last_n_iterations=last_n_iterations,
    )

    # Récompenses de l'équipe
    plot_all_team_rewards(
        logs,
        rewards_directory / "all_team_rewards.png",
        window_size=reward_window_size,
    )

    # Figures regroupées 2 x 2 pour le chapitre du mémoire
    plot_grouped_reward_details(
        logs,
        rewards_directory
        / "grouped_reward_details.png",
        window_size=reward_window_size,
    )

    plot_grouped_pairwise_reward_comparisons(
        logs,
        rewards_directory
        / "grouped_pairwise_comparisons.png",
        window_size=reward_window_size,
    )

    plot_final_training_performance(
        logs,
        rewards_directory
        / "final_training_performance.png",
        last_n_iterations=last_n_iterations,
    )

    plot_final_reward_boxplot(
        logs,
        rewards_directory
        / "final_reward_boxplot.png",
        last_n_iterations=last_n_iterations,
    )

    # Pertes
    plot_average_loss_comparison(
        logs,
        losses_directory
        / "average_loss_comparison.png",
        window_size=loss_window_size,
    )

    plot_grouped_agent_losses(
        logs,
        losses_directory,
        window_size=loss_window_size,
    )

    plot_final_agent_loss_bars(
        logs,
        losses_directory,
        last_n_iterations=last_n_iterations,
    )

    # Temps
    plot_reward_vs_cumulative_time(
        logs,
        timing_directory
        / "reward_vs_cumulative_time.png",
        window_size=reward_window_size,
    )

    plot_total_time_comparison(
        logs,
        timing_directory
        / "total_time_comparison.png",
    )

    # Tableaux
    export_training_summary_csv(
        logs,
        tables_directory
        / "training_summary.csv",
        last_n_iterations=last_n_iterations,
    )

    export_agent_loss_summary_csv(
        logs,
        tables_directory
        / "agent_loss_summary.csv",
        last_n_iterations=last_n_iterations,
    )

    print(
        "\n[INFO] Génération terminée. "
        f"Résultats disponibles dans : {output_directory}"
    )


def main() -> None:
    logs = load_logs("logs")

    generate_result_figures(
        logs=logs,
        output_directory="plot_results",
        reward_window_size=10,
        loss_window_size=10,
        last_n_iterations=50,
    )


if __name__ == "__main__":
    main()
