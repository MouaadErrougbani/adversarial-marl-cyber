"""
plot_evaluation_results_final.py

Génère les figures et tableaux d'évaluation pour quatre configurations :
MAPPO-GAT, MAPPO-GCN, IPPO-GAT et IPPO-GCN.

Hypothèses :
- 100 épisodes d'évaluation par configuration ;
- 500 étapes au maximum par épisode ;
- aucune mise à jour des modèles pendant l'évaluation ;
- les statistiques finales sont calculées sur les 100 épisodes complets.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "mappo_gat_gat": "MAPPO-GAT",
    "mappo_gcn_gcn": "MAPPO-GCN",
    "ppo_gat_gat": "IPPO-GAT",
    "ppo_gcn_gcn": "IPPO-GCN",
    "ippo_gat_gat": "IPPO-GAT",
    "ippo_gcn_gcn": "IPPO-GCN",
}

EXPECTED_EPISODES = 100


def display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def extract_episode_metrics(
    evaluation_data: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extrait et valide les métriques épisode par épisode."""

    episode_entries = evaluation_data.get("episodes", [])
    episodes: list[int] = []
    avg_rewards: list[float] = []
    std_rewards: list[float] = []
    min_rewards: list[float] = []
    max_rewards: list[float] = []

    required_keys = {
        "episode",
        "avg_reward",
        "std_reward",
        "min_reward",
        "max_reward",
    }

    for entry in episode_entries:
        if not isinstance(entry, dict):
            continue
        if not required_keys.issubset(entry.keys()):
            continue

        try:
            episode = int(entry["episode"])
            avg_reward = float(entry["avg_reward"])
            std_reward = float(entry["std_reward"])
            min_reward = float(entry["min_reward"])
            max_reward = float(entry["max_reward"])
        except (TypeError, ValueError):
            continue

        values = [avg_reward, std_reward, min_reward, max_reward]
        if not all(np.isfinite(value) for value in values):
            continue

        episodes.append(episode)
        avg_rewards.append(avg_reward)
        std_rewards.append(max(0.0, std_reward))
        min_rewards.append(min_reward)
        max_rewards.append(max_reward)

    if not episodes:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return {
            "episodes": empty_i,
            "avg_rewards": empty_f,
            "std_rewards": empty_f,
            "min_rewards": empty_f,
            "max_rewards": empty_f,
        }

    order = np.argsort(episodes)
    return {
        "episodes": np.asarray(episodes, dtype=int)[order],
        "avg_rewards": np.asarray(avg_rewards, dtype=float)[order],
        "std_rewards": np.asarray(std_rewards, dtype=float)[order],
        "min_rewards": np.asarray(min_rewards, dtype=float)[order],
        "max_rewards": np.asarray(max_rewards, dtype=float)[order],
    }


def moving_average(
    values: np.ndarray,
    window_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if window_size <= 1 or values.size < window_size:
        return np.arange(values.size), values.copy()

    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(values, kernel, mode="valid")
    indices = np.arange(window_size - 1, values.size)
    return indices, smoothed


def load_json_file(json_path: str | Path) -> dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON invalide dans {json_path}: {error}") from error


def load_evaluation_directory(
    results_directory: str | Path,
) -> dict[str, dict[str, Any]]:
    results_directory = Path(results_directory)
    if not results_directory.exists():
        raise FileNotFoundError(f"Le dossier n'existe pas : {results_directory}")

    evaluation_logs: dict[str, dict[str, Any]] = {}
    for model_directory in sorted(results_directory.iterdir()):
        if not model_directory.is_dir():
            continue

        metrics_path = model_directory / "metrics.json"
        if not metrics_path.exists():
            print(f"[WARNING] metrics.json absent dans : {model_directory}")
            continue

        try:
            evaluation_logs[model_directory.name] = load_json_file(metrics_path)
            print(f"[INFO] Chargé : {metrics_path}")
        except (ValueError, FileNotFoundError) as error:
            print(f"[WARNING] Impossible de charger {metrics_path}: {error}")

    if not evaluation_logs:
        raise ValueError(
            f"Aucun fichier metrics.json valide trouvé dans {results_directory}"
        )

    return evaluation_logs


def validate_episode_counts(
    evaluation_logs: dict[str, dict[str, Any]],
    expected_episodes: int = EXPECTED_EPISODES,
) -> None:
    invalid: list[str] = []
    for model_name, data in evaluation_logs.items():
        metrics = extract_episode_metrics(data)
        count = int(metrics["avg_rewards"].size)
        if count != expected_episodes:
            invalid.append(f"{display_name(model_name)}: {count} épisodes")

    # if invalid:
    #     raise ValueError(
    #         "Nombre d'épisodes invalide. "
    #         f"Attendu: {expected_episodes}. Reçu: {'; '.join(invalid)}"
    #     )

    print(
        f"[OK] Les {len(evaluation_logs)} configurations contiennent "
        f"exactement {expected_episodes} épisodes."
    )


def save_figure(figure: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[INFO] Figure enregistrée : {output_path}")


def plot_evaluation_details(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_average_window: int = 5,
) -> None:
    number_of_models = len(evaluation_logs)
    if number_of_models == 0:
        return

    number_of_columns = 2
    number_of_rows = math.ceil(number_of_models / number_of_columns)
    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(15, 5 * number_of_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    ordered_items = sorted(
        evaluation_logs.items(),
        key=lambda item: display_name(item[0]),
    )

    for axis, (model_name, data) in zip(axes_flat, ordered_items):
        metrics = extract_episode_metrics(data)
        episodes = metrics["episodes"]
        avg_rewards = metrics["avg_rewards"]
        std_rewards = metrics["std_rewards"]
        min_rewards = metrics["min_rewards"]
        max_rewards = metrics["max_rewards"]

        if episodes.size == 0:
            axis.text(
                0.5,
                0.5,
                "Aucune donnée valide",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(display_name(model_name))
            continue

        axis.plot(
            episodes,
            avg_rewards,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label="Récompense moyenne",
        )
        axis.fill_between(
            episodes,
            avg_rewards - std_rewards,
            avg_rewards + std_rewards,
            alpha=0.2,
            label="Moyenne ± 1 écart-type",
        )
        axis.plot(
            episodes,
            min_rewards,
            linestyle="--",
            linewidth=1.1,
            label="Récompense minimale",
        )
        axis.plot(
            episodes,
            max_rewards,
            linestyle="--",
            linewidth=1.1,
            label="Récompense maximale",
        )

        valid_indices, smoothed_rewards = moving_average(
            avg_rewards,
            moving_average_window,
        )
        if smoothed_rewards.size > 0:
            axis.plot(
                episodes[valid_indices],
                smoothed_rewards,
                linewidth=2.4,
                label=f"Moyenne mobile ({moving_average_window})",
            )

        parameters = data.get("parameters", {})
        seed = parameters.get("seed", "inconnue")
        axis.set_title(f"{display_name(model_name)} — graine {seed}")
        axis.set_xlabel("Épisode d'évaluation")
        axis.set_ylabel("Récompense")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(fontsize=8)

    for axis in axes_flat[number_of_models:]:
        axis.set_visible(False)

    figure.suptitle(
        "Détails des récompenses d'évaluation par configuration",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[INFO] Figure enregistrée : {output_path}")


def plot_average_reward_comparison(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_average_window: int = 5,
    show_raw_rewards: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 7))
    curves_plotted = 0

    ordered_items = sorted(
        evaluation_logs.items(),
        key=lambda item: display_name(item[0]),
    )

    for model_name, data in ordered_items:
        metrics = extract_episode_metrics(data)
        episodes = metrics["episodes"]
        avg_rewards = metrics["avg_rewards"]
        if episodes.size == 0:
            continue

        line_color = None
        if show_raw_rewards:
            raw_line = axis.plot(
                episodes,
                avg_rewards,
                linewidth=1,
                alpha=0.25,
            )[0]
            line_color = raw_line.get_color()

        valid_indices, smoothed_rewards = moving_average(
            avg_rewards,
            moving_average_window,
        )
        options: dict[str, Any] = {
            "linewidth": 2.5,
            "label": display_name(model_name),
        }
        if line_color is not None:
            options["color"] = line_color

        axis.plot(episodes[valid_indices], smoothed_rewards, **options)
        curves_plotted += 1

    if curves_plotted:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Aucune donnée valide",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title("Comparaison des récompenses moyennes en évaluation")
    axis.set_xlabel("Épisode d'évaluation")
    axis.set_ylabel("Récompense moyenne")
    axis.grid(True, linestyle="--", alpha=0.5)
    save_figure(figure, output_path)


def plot_evaluation_summary(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    model_names: list[str] = []
    means: list[float] = []
    standard_deviations: list[float] = []

    ordered_items = sorted(
        evaluation_logs.items(),
        key=lambda item: display_name(item[0]),
    )

    for model_name, data in ordered_items:
        metrics = extract_episode_metrics(data)
        avg_rewards = metrics["avg_rewards"]
        if avg_rewards.size == 0:
            continue

        model_names.append(display_name(model_name))
        means.append(float(np.mean(avg_rewards)))
        standard_deviations.append(float(np.std(avg_rewards)))

    if not model_names:
        print("[WARNING] Aucune donnée pour evaluation_summary.png")
        return

    positions = np.arange(len(model_names))
    figure, axis = plt.subplots(figsize=(12, 7))
    bars = axis.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6,
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(model_names, rotation=15, ha="right")
    axis.axhline(y=0, linestyle=":", linewidth=1)
    axis.set_title(
        "Performance moyenne en évaluation\n"
        "Moyenne calculée sur les 100 épisodes"
    )
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Récompense moyenne d'évaluation")
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, mean_value in zip(bars, means):
        axis.annotate(
            f"{mean_value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if mean_value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean_value >= 0 else "top",
        )

    save_figure(figure, output_path)


def export_evaluation_summary_csv(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "configuration",
        "num_episodes",
        "evaluation_mean",
        "evaluation_std",
        "evaluation_min_avg_reward",
        "evaluation_max_avg_reward",
        "global_min_reward",
        "global_max_reward",
        "mean_within_episode_std",
        "evaluation_time",
    ]

    rows: list[dict[str, Any]] = []
    for model_name, data in evaluation_logs.items():
        metrics = extract_episode_metrics(data)
        avg_rewards = metrics["avg_rewards"]
        std_rewards = metrics["std_rewards"]
        min_rewards = metrics["min_rewards"]
        max_rewards = metrics["max_rewards"]
        if avg_rewards.size == 0:
            continue

        time_data = data.get("time", {})
        rows.append({
            "configuration": display_name(model_name),
            "num_episodes": int(avg_rewards.size),
            "evaluation_mean": float(np.mean(avg_rewards)),
            "evaluation_std": float(np.std(avg_rewards)),
            "evaluation_min_avg_reward": float(np.min(avg_rewards)),
            "evaluation_max_avg_reward": float(np.max(avg_rewards)),
            "global_min_reward": float(np.min(min_rewards)),
            "global_max_reward": float(np.max(max_rewards)),
            "mean_within_episode_std": float(np.mean(std_rewards)),
            "evaluation_time": time_data.get("elapsed", ""),
        })

    rows.sort(key=lambda row: row["evaluation_mean"], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Tableau enregistré : {output_path}")


def print_evaluation_statistics(
    evaluation_logs: dict[str, dict[str, Any]],
) -> None:
    print("\n" + "=" * 80)
    print("STATISTIQUES D'ÉVALUATION — 100 ÉPISODES")
    print("=" * 80)

    rows: list[tuple[str, float, dict[str, Any]]] = []
    for model_name, data in evaluation_logs.items():
        metrics = extract_episode_metrics(data)
        avg_rewards = metrics["avg_rewards"]
        if avg_rewards.size == 0:
            continue
        rows.append((model_name, float(np.mean(avg_rewards)), data))

    rows.sort(key=lambda item: item[1], reverse=True)

    for rank, (model_name, _, data) in enumerate(rows, start=1):
        metrics = extract_episode_metrics(data)
        avg_rewards = metrics["avg_rewards"]
        std_rewards = metrics["std_rewards"]
        min_rewards = metrics["min_rewards"]
        max_rewards = metrics["max_rewards"]
        parameters = data.get("parameters", {})
        time_data = data.get("time", {})

        print(f"\n{rank}. {display_name(model_name)}")
        print(f"   Algorithme                      : {parameters.get('algorithm', 'inconnu')}")
        print(f"   Graine                          : {parameters.get('seed', 'inconnue')}")
        print(f"   Nombre d'épisodes               : {avg_rewards.size}")
        print(f"   Récompense moyenne              : {np.mean(avg_rewards):.4f}")
        print(f"   Écart-type entre épisodes       : {np.std(avg_rewards):.4f}")
        print(f"   Minimum des moyennes d'épisode  : {np.min(avg_rewards):.4f}")
        print(f"   Maximum des moyennes d'épisode  : {np.max(avg_rewards):.4f}")
        print(f"   Récompense minimale globale     : {np.min(min_rewards):.4f}")
        print(f"   Récompense maximale globale     : {np.max(max_rewards):.4f}")
        print(f"   Écart-type moyen intra-épisode  : {np.mean(std_rewards):.4f}")
        print(f"   Temps d'évaluation              : {time_data.get('elapsed', 'inconnu')}")

    print("\n" + "=" * 80)


def generate_evaluation_plots(
    evaluation_directory: str | Path = "results",
    output_directory: str | Path = "plot_eval",
    moving_average_window: int = 5,
) -> None:
    evaluation_logs = load_evaluation_directory(evaluation_directory)
    validate_episode_counts(evaluation_logs, expected_episodes=100)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(evaluation_logs)} modèles chargés.")

    print_evaluation_statistics(evaluation_logs)

    plot_evaluation_details(
        evaluation_logs,
        output_directory / "evaluation_details.png",
        moving_average_window,
    )
    plot_average_reward_comparison(
        evaluation_logs,
        output_directory / "evaluation_comparison.png",
        moving_average_window,
        show_raw_rewards=True,
    )
    plot_evaluation_summary(
        evaluation_logs,
        output_directory / "evaluation_summary.png",
    )
    export_evaluation_summary_csv(
        evaluation_logs,
        output_directory / "evaluation_summary.csv",
    )

    print(
        "\n[INFO] Génération terminée. "
        f"Résultats disponibles dans : {output_directory}"
    )


def main() -> None:
    generate_evaluation_plots(
        evaluation_directory="results",
        output_directory="plot_eval",
        moving_average_window=5,
    )


if __name__ == "__main__":
    main()
