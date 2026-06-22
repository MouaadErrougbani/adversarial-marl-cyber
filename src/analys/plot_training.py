"""
plot_training.py

Comparaison correcte des entraînements :

- MAPPO vs IPPO avec GAT fixé
- MAPPO vs IPPO avec GCN fixé
- GAT vs GCN avec MAPPO fixé
- GAT vs GCN avec IPPO fixé

Les fichiers attendus sont :

logs/
├── mappo_gat_gat.pt
├── mappo_gcn_gcn.pt
├── ppo_gat_gat.pt
└── ppo_gcn_gcn.pt

Dans ce script, les fichiers commençant par "ppo" sont affichés
comme IPPO, car chaque agent possède sa propre politique/critic.
"""

import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Chargement et utilitaires
# ============================================================

def load_logs(log_directory: str | Path) -> dict[str, list[dict[str, Any]]]:
    """
    Charge tous les fichiers .pt du dossier de logs.
    """

    log_directory = Path(log_directory)

    if not log_directory.exists():
        raise FileNotFoundError(
            f"Le dossier de logs n'existe pas : {log_directory}"
        )

    logs: dict[str, list[dict[str, Any]]] = {}

    for file_path in sorted(log_directory.glob("*.pt")):
        model_name = file_path.stem

        history = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False
        )

        if not isinstance(history, list):
            print(
                f"[WARNING] {file_path} ignoré : "
                "le contenu n'est pas une liste."
            )
            continue

        logs[model_name] = history

        print(
            f"[INFO] {model_name}: "
            f"{len(history)} itérations chargées"
        )

    if not logs:
        raise ValueError(
            f"Aucun fichier .pt valide trouvé dans {log_directory}"
        )

    return logs


def display_name(model_name: str) -> str:
    """
    Transforme le nom interne en nom lisible.

    Exemples :
        mappo_gat_gat -> MAPPO-GAT
        ppo_gcn_gcn   -> IPPO-GCN
    """

    name = model_name.lower()

    algorithm = "MAPPO" if name.startswith("mappo") else "IPPO"

    if "gat" in name:
        architecture = "GAT"
    elif "gcn" in name:
        architecture = "GCN"
    else:
        architecture = "Unknown"

    return f"{algorithm}-{architecture}"


def extract_metric(
    history: list[dict[str, Any]],
    metric_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait une métrique en supprimant les valeurs invalides.

    L'axe x conserve les vraies positions dans l'historique.
    """

    iterations: list[int] = []
    values: list[float] = []

    for index, step in enumerate(history):
        value = step.get(metric_name)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value):
            continue

        iterations.append(index)
        values.append(value)

    return (
        np.asarray(iterations, dtype=int),
        np.asarray(values, dtype=float)
    )


def moving_average(
    values: np.ndarray,
    window_size: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule une moyenne mobile.
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    if window_size <= 1 or values.size < window_size:
        return np.arange(values.size), values.copy()

    kernel = np.ones(window_size) / window_size

    smoothed = np.convolve(
        values,
        kernel,
        mode="valid"
    )

    indices = np.arange(
        window_size - 1,
        values.size
    )

    return indices, smoothed


def get_selected_logs(
    logs: dict[str, list[dict[str, Any]]],
    algorithm: str | None = None,
    architecture: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """
    Sélectionne les modèles selon l'algorithme ou l'architecture.

    algorithm :
        "mappo" ou "ippo"

    architecture :
        "gat" ou "gcn"
    """

    selected = {}

    for model_name, history in logs.items():
        lower_name = model_name.lower()

        if algorithm == "mappo" and not lower_name.startswith("mappo"):
            continue

        if algorithm == "ippo" and not (
            lower_name.startswith("ppo")
            or lower_name.startswith("ippo")
        ):
            continue

        if architecture is not None and architecture not in lower_name:
            continue

        selected[model_name] = history

    return selected


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Crée le dossier parent d'un fichier.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 1. Figure détaillée, similaire à l'évaluation
# ============================================================

def plot_training_reward_details(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 5
) -> None:
    """
    Crée un sous-graphe par configuration.

    Chaque sous-graphe contient :
    - avg_reward ;
    - avg_reward ± std_reward ;
    - min_reward ;
    - max_reward ;
    - moyenne mobile.
    """

    number_of_models = len(logs)

    if number_of_models == 0:
        return

    number_of_columns = 2
    number_of_rows = math.ceil(
        number_of_models / number_of_columns
    )

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(16, 5.5 * number_of_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for axis, (model_name, history) in zip(
        axes,
        logs.items()
    ):
        _, avg_rewards = extract_metric(
            history,
            "avg_reward"
        )

        _, std_rewards = extract_metric(
            history,
            "std_reward"
        )

        _, min_rewards = extract_metric(
            history,
            "min_reward"
        )

        _, max_rewards = extract_metric(
            history,
            "max_reward"
        )

        common_length = min(
            len(avg_rewards),
            len(std_rewards),
            len(min_rewards),
            len(max_rewards)
        )

        if common_length == 0:
            axis.text(
                0.5,
                0.5,
                "Aucune donnée de reward",
                ha="center",
                va="center",
                transform=axis.transAxes
            )
            axis.set_title(display_name(model_name))
            continue

        avg_rewards = avg_rewards[:common_length]
        std_rewards = std_rewards[:common_length]
        min_rewards = min_rewards[:common_length]
        max_rewards = max_rewards[:common_length]

        iterations = np.arange(common_length)

        axis.plot(
            iterations,
            avg_rewards,
            marker="o",
            markersize=3,
            linewidth=1.3,
            alpha=0.75,
            label="Average reward"
        )

        axis.fill_between(
            iterations,
            avg_rewards - std_rewards,
            avg_rewards + std_rewards,
            alpha=0.2,
            label="Average ± 1 std"
        )

        axis.plot(
            iterations,
            min_rewards,
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label="Minimum reward"
        )

        axis.plot(
            iterations,
            max_rewards,
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label="Maximum reward"
        )

        smooth_indices, smooth_rewards = moving_average(
            avg_rewards,
            window_size=window_size
        )

        axis.plot(
            iterations[smooth_indices],
            smooth_rewards,
            linewidth=2.5,
            label=f"Moving average (window={window_size})"
        )

        axis.axhline(
            y=0,
            linestyle=":",
            linewidth=1,
            alpha=0.7
        )

        axis.set_title(
            f"Training - {display_name(model_name)}"
        )
        axis.set_xlabel("Training iteration")
        axis.set_ylabel("Reward")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(fontsize=8)

    for axis in axes[number_of_models:]:
        axis.set_visible(False)

    figure.suptitle(
        "Training Rewards per Configuration",
        fontsize=17
    )

    figure.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = ensure_parent_directory(save_path)

    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {save_path}")


# ============================================================
# Fonction générique de comparaison
# ============================================================

def plot_reward_comparison(
    selected_logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    title: str,
    window_size: int = 5
) -> None:
    """
    Compare les récompenses de modèles compatibles.

    Les courbes fines représentent les valeurs brutes.
    Les courbes épaisses représentent la moyenne mobile.
    """

    figure, axis = plt.subplots(figsize=(12, 7))

    plotted_curves = 0

    for model_name, history in selected_logs.items():
        iterations, rewards = extract_metric(
            history,
            "avg_reward"
        )

        if rewards.size == 0:
            continue

        raw_line = axis.plot(
            iterations,
            rewards,
            linewidth=1,
            alpha=0.25
        )[0]

        smooth_indices, smooth_rewards = moving_average(
            rewards,
            window_size=window_size
        )

        axis.plot(
            iterations[smooth_indices],
            smooth_rewards,
            linewidth=2.8,
            color=raw_line.get_color(),
            label=display_name(model_name)
        )

        plotted_curves += 1

    if plotted_curves == 0:
        axis.text(
            0.5,
            0.5,
            "Aucune donnée disponible",
            ha="center",
            va="center",
            transform=axis.transAxes
        )
    else:
        axis.legend()

    axis.axhline(
        y=0,
        linestyle=":",
        linewidth=1,
        alpha=0.7
    )

    axis.set_title(title)
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Average reward")
    axis.grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()

    save_path = ensure_parent_directory(save_path)

    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {save_path}")


# ============================================================
# 2. Comparaison MAPPO contre IPPO
# ============================================================

def plot_algorithm_comparisons(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 5
) -> None:
    """
    Compare MAPPO et IPPO en fixant l'architecture.
    """

    output_directory = Path(output_directory)

    gat_logs = get_selected_logs(
        logs,
        architecture="gat"
    )

    plot_reward_comparison(
        selected_logs=gat_logs,
        save_path=output_directory / "mappo_vs_ippo_gat.png",
        title="MAPPO vs IPPO — GAT Architecture",
        window_size=window_size
    )

    gcn_logs = get_selected_logs(
        logs,
        architecture="gcn"
    )

    plot_reward_comparison(
        selected_logs=gcn_logs,
        save_path=output_directory / "mappo_vs_ippo_gcn.png",
        title="MAPPO vs IPPO — GCN Architecture",
        window_size=window_size
    )


# ============================================================
# 3. Comparaison GAT contre GCN
# ============================================================

def plot_architecture_comparisons(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    window_size: int = 5
) -> None:
    """
    Compare GAT et GCN en fixant l'algorithme.
    """

    output_directory = Path(output_directory)

    mappo_logs = get_selected_logs(
        logs,
        algorithm="mappo"
    )

    plot_reward_comparison(
        selected_logs=mappo_logs,
        save_path=output_directory / "gat_vs_gcn_mappo.png",
        title="GAT vs GCN — MAPPO",
        window_size=window_size
    )

    ippo_logs = get_selected_logs(
        logs,
        algorithm="ippo"
    )

    plot_reward_comparison(
        selected_logs=ippo_logs,
        save_path=output_directory / "gat_vs_gcn_ippo.png",
        title="GAT vs GCN — IPPO",
        window_size=window_size
    )


# ============================================================
# 4. Résumé final
# ============================================================

def plot_final_training_performance(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    last_n_iterations: int = 10
) -> None:
    """
    Compare les performances sur les dernières itérations.

    La hauteur de la barre représente la moyenne de avg_reward.
    La barre d'erreur représente l'écart-type entre les dernières
    valeurs de avg_reward.
    """

    names: list[str] = []
    means: list[float] = []
    standard_deviations: list[float] = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(
            history,
            "avg_reward"
        )

        if rewards.size == 0:
            continue

        number_of_values = min(
            last_n_iterations,
            rewards.size
        )

        final_rewards = rewards[-number_of_values:]

        names.append(display_name(model_name))
        means.append(float(np.mean(final_rewards)))
        standard_deviations.append(
            float(np.std(final_rewards))
        )

    if not names:
        return

    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(11, 7))

    bars = axis.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        names,
        rotation=15,
        ha="right"
    )

    axis.axhline(
        y=0,
        linestyle=":",
        linewidth=1
    )

    axis.set_title(
        "Final Training Performance\n"
        f"Mean over the last {last_n_iterations} iterations"
    )
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Mean training reward")
    axis.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    for bar, mean_value in zip(bars, means):
        axis.annotate(
            f"{mean_value:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(
                0,
                4 if mean_value >= 0 else -15
            ),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean_value >= 0 else "top"
        )

    figure.tight_layout()

    save_path = ensure_parent_directory(save_path)

    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {save_path}")


# ============================================================
# 5. Efficacité par rapport au temps
# ============================================================

def plot_reward_vs_cumulative_time(
    logs: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_size: int = 5
) -> None:
    """
    Compare la récompense selon le temps cumulé.

    Le temps cumulé inclut :
    - training_time_sec ;
    - collection_time_sec.

    Cette figure est plus pertinente qu'un scatter plot
    reward contre temps d'une seule itération.
    """

    figure, axis = plt.subplots(figsize=(12, 7))

    curves_plotted = 0

    for model_name, history in logs.items():
        rewards: list[float] = []
        iteration_times: list[float] = []

        for step in history:
            reward = step.get("avg_reward")

            training_time = step.get(
                "training_time_sec",
                0.0
            )

            collection_time = step.get(
                "collection_time_sec",
                0.0
            )

            try:
                reward = float(reward)
                total_time = (
                    float(training_time)
                    + float(collection_time)
                )
            except (TypeError, ValueError):
                continue

            if not (
                np.isfinite(reward)
                and np.isfinite(total_time)
            ):
                continue

            rewards.append(reward)
            iteration_times.append(max(0.0, total_time))

        if not rewards:
            continue

        rewards_array = np.asarray(rewards)
        cumulative_time = np.cumsum(iteration_times)

        smooth_indices, smooth_rewards = moving_average(
            rewards_array,
            window_size=window_size
        )

        axis.plot(
            cumulative_time[smooth_indices] / 60.0,
            smooth_rewards,
            linewidth=2.5,
            label=display_name(model_name)
        )

        curves_plotted += 1

    if curves_plotted > 0:
        axis.legend()

    axis.set_title(
        "Training Efficiency — Reward vs Cumulative Time"
    )
    axis.set_xlabel("Cumulative time (minutes)")
    axis.set_ylabel("Average reward")
    axis.grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()

    save_path = ensure_parent_directory(save_path)

    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {save_path}")


# ============================================================
# Statistiques textuelles
# ============================================================

def print_training_summary(
    logs: dict[str, list[dict[str, Any]]],
    last_n_iterations: int = 10
) -> None:
    """
    Affiche un résumé numérique dans le terminal.
    """

    print("\n" + "=" * 78)
    print("TRAINING PERFORMANCE SUMMARY")
    print("=" * 78)

    results = []

    for model_name, history in logs.items():
        _, rewards = extract_metric(
            history,
            "avg_reward"
        )

        if rewards.size == 0:
            continue

        number_of_values = min(
            last_n_iterations,
            rewards.size
        )

        final_rewards = rewards[-number_of_values:]

        total_time = 0.0

        for step in history:
            try:
                total_time += float(
                    step.get("training_time_sec", 0.0)
                )
                total_time += float(
                    step.get("collection_time_sec", 0.0)
                )
            except (TypeError, ValueError):
                continue

        result = {
            "name": display_name(model_name),
            "iterations": rewards.size,
            "global_mean": float(np.mean(rewards)),
            "final_mean": float(np.mean(final_rewards)),
            "final_std": float(np.std(final_rewards)),
            "best": float(np.max(rewards)),
            "total_minutes": total_time / 60.0,
        }

        results.append(result)

    results.sort(
        key=lambda item: item["final_mean"],
        reverse=True
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['name']}")
        print(
            f"   Iterations       : {result['iterations']}"
        )
        print(
            f"   Global mean      : "
            f"{result['global_mean']:.4f}"
        )
        print(
            f"   Final mean       : "
            f"{result['final_mean']:.4f}"
        )
        print(
            f"   Final std        : "
            f"{result['final_std']:.4f}"
        )
        print(
            f"   Best reward      : "
            f"{result['best']:.4f}"
        )
        print(
            f"   Total time (min) : "
            f"{result['total_minutes']:.2f}"
        )

    print("\n" + "=" * 78)


# ============================================================
# Génération des graphiques utiles
# ============================================================

def generate_training_plots(
    logs: dict[str, list[dict[str, Any]]],
    output_directory: str | Path = "plot_training",
    window_size: int = 5,
    last_n_iterations: int = 10
) -> None:
    """
    Génère uniquement les graphiques utiles pour l'analyse.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print_training_summary(
        logs,
        last_n_iterations=last_n_iterations
    )

    plot_training_reward_details(
        logs,
        save_path=(
            output_directory
            / "training_reward_details.png"
        ),
        window_size=window_size
    )

    plot_algorithm_comparisons(
        logs,
        output_directory=output_directory,
        window_size=window_size
    )

    plot_architecture_comparisons(
        logs,
        output_directory=output_directory,
        window_size=window_size
    )

    plot_final_training_performance(
        logs,
        save_path=(
            output_directory
            / "final_training_performance.png"
        ),
        last_n_iterations=last_n_iterations
    )

    plot_reward_vs_cumulative_time(
        logs,
        save_path=(
            output_directory
            / "reward_vs_cumulative_time.png"
        ),
        window_size=window_size
    )

    print(
        f"\n[INFO] Tous les graphiques sont dans : "
        f"{output_directory}"
    )


# ============================================================
# Exécution
# ============================================================

def main() -> None:
    logs = load_logs("logs")

    generate_training_plots(
        logs=logs,
        output_directory="plot_training",
        window_size=5,
        last_n_iterations=10
    )


if __name__ == "__main__":
    main()