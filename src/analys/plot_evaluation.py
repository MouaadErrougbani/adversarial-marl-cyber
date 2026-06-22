from pathlib import Path
import json
from typing import Any

from pathlib import Path
import json
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

def extract_episode_metrics(
    evaluation_data: dict[str, Any]
) -> dict[str, np.ndarray]:
    """
    Extrait et valide les métriques des épisodes.
    """

    episode_entries = evaluation_data.get("episodes", [])

    episodes = []
    avg_rewards = []
    std_rewards = []
    min_rewards = []
    max_rewards = []

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

        values = [
            avg_reward,
            std_reward,
            min_reward,
            max_reward,
        ]

        if not all(np.isfinite(value) for value in values):
            continue

        episodes.append(episode)
        avg_rewards.append(avg_reward)
        std_rewards.append(max(0.0, std_reward))
        min_rewards.append(min_reward)
        max_rewards.append(max_reward)

    if not episodes:
        return {
            "episodes": np.array([], dtype=int),
            "avg_rewards": np.array([], dtype=float),
            "std_rewards": np.array([], dtype=float),
            "min_rewards": np.array([], dtype=float),
            "max_rewards": np.array([], dtype=float),
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
    window_size: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule une moyenne mobile.

    Retourne les indices correspondants et les valeurs lissées.
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    if window_size <= 1 or values.size < window_size:
        return np.arange(values.size), values.copy()

    kernel = np.ones(window_size) / window_size

    smoothed_values = np.convolve(
        values,
        kernel,
        mode="valid"
    )

    valid_indices = np.arange(
        window_size - 1,
        values.size
    )

    return valid_indices, smoothed_values

def plot_evaluation_details(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_average_window: int = 5
) -> None:
    """
    Crée une figure composée d'un sous-graphe par modèle.
    """

    number_of_models = len(evaluation_logs)

    if number_of_models == 0:
        return

    number_of_columns = 2
    number_of_rows = math.ceil(
        number_of_models / number_of_columns
    )

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(15, 5 * number_of_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for axis, (model_name, data) in zip(
        axes,
        evaluation_logs.items()
    ):
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
                transform=axis.transAxes
            )

            axis.set_title(model_name)
            continue

        axis.plot(
            episodes,
            avg_rewards,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Average reward"
        )

        axis.fill_between(
            episodes,
            avg_rewards - std_rewards,
            avg_rewards + std_rewards,
            alpha=0.2,
            label="Average ± 1 std"
        )

        axis.plot(
            episodes,
            min_rewards,
            linestyle="--",
            linewidth=1.2,
            label="Minimum reward"
        )

        axis.plot(
            episodes,
            max_rewards,
            linestyle="--",
            linewidth=1.2,
            label="Maximum reward"
        )

        valid_indices, smoothed_rewards = moving_average(
            avg_rewards,
            window_size=moving_average_window
        )

        if smoothed_rewards.size > 0:
            axis.plot(
                episodes[valid_indices],
                smoothed_rewards,
                linewidth=2.5,
                label=(
                    f"Moving average "
                    f"(window={moving_average_window})"
                )
            )

        parameters = data.get("parameters", {})

        algorithm = parameters.get(
            "algorithm",
            "Unknown"
        )

        seed = parameters.get(
            "seed",
            "Unknown"
        )

        axis.set_title(
            f"Evaluation - {model_name}\n"
            f"{algorithm}, seed={seed}"
        )

        axis.set_xlabel("Evaluation Episode")
        axis.set_ylabel("Reward")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(fontsize=8)

    for axis in axes[number_of_models:]:
        axis.set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.suptitle(
        "Evaluation Rewards per Algorithm",
        fontsize=16
    )

    figure.tight_layout(rect=[0, 0, 1, 0.96])

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {output_path}")

def plot_evaluation_summary(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
    last_n_episodes: int = 10
) -> None:
    """
    Compare la récompense moyenne obtenue pendant
    les derniers épisodes de chaque modèle.
    """

    model_names = []
    final_means = []
    final_stds = []

    for model_name, data in evaluation_logs.items():

        metrics = extract_episode_metrics(data)
        avg_rewards = metrics["avg_rewards"]

        if avg_rewards.size == 0:
            continue

        number_of_episodes = min(
            last_n_episodes,
            avg_rewards.size
        )

        selected_rewards = avg_rewards[
            -number_of_episodes:
        ]

        model_names.append(model_name)
        final_means.append(
            float(np.mean(selected_rewards))
        )
        final_stds.append(
            float(np.std(selected_rewards))
        )

    if not model_names:
        print(
            "[WARNING] Aucune donnée pour "
            "evaluation_summary.png"
        )
        return

    positions = np.arange(len(model_names))

    figure, axis = plt.subplots(figsize=(12, 7))

    bars = axis.bar(
        positions,
        final_means,
        yerr=final_stds,
        capsize=6
    )

    axis.set_xticks(positions)

    axis.set_xticklabels(
        model_names,
        rotation=20,
        ha="right"
    )

    axis.axhline(
        y=0,
        linestyle=":",
        linewidth=1
    )

    axis.set_title(
        "Final Evaluation Performance\n"
        f"Mean over the last {last_n_episodes} episodes"
    )

    axis.set_xlabel("Model")
    axis.set_ylabel("Mean Evaluation Reward")
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, mean_value in zip(bars, final_means):
        axis.annotate(
            f"{mean_value:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(
                0,
                4 if mean_value >= 0 else -14
            ),
            textcoords="offset points",
            ha="center",
            va=(
                "bottom"
                if mean_value >= 0
                else "top"
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {output_path}")

def plot_average_reward_comparison(
    evaluation_logs: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_average_window: int = 5,
    show_raw_rewards: bool = True
) -> None:
    """
    Compare les récompenses moyennes des modèles
    dans une seule figure.
    """

    figure, axis = plt.subplots(figsize=(13, 7))

    curves_plotted = 0

    for model_name, data in evaluation_logs.items():

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
                alpha=0.25
            )[0]

            line_color = raw_line.get_color()

        valid_indices, smoothed_rewards = moving_average(
            avg_rewards,
            window_size=moving_average_window
        )

        plot_options = {
            "linewidth": 2.5,
            "label": model_name,
        }

        if line_color is not None:
            plot_options["color"] = line_color

        axis.plot(
            episodes[valid_indices],
            smoothed_rewards,
            **plot_options
        )

        curves_plotted += 1

    if curves_plotted > 0:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Aucune donnée valide",
            ha="center",
            va="center",
            transform=axis.transAxes
        )

    axis.axhline(
        y=0,
        linestyle=":",
        linewidth=1
    )

    axis.set_title(
        "Average Evaluation Reward — Algorithm Comparison"
    )

    axis.set_xlabel("Evaluation Episode")
    axis.set_ylabel("Average Reward")
    axis.grid(True, linestyle="--", alpha=0.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"[INFO] Figure enregistrée : {output_path}")

def load_json_file(json_path: str | Path) -> dict[str, Any]:
    """
    Charge un fichier JSON.
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Le fichier n'existe pas : {json_path}"
        )

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON invalide dans {json_path}: {error}"
        ) from error

    return data


def load_evaluation_directory(
    results_directory: str | Path
) -> dict[str, dict[str, Any]]:
    """
    Cherche automatiquement les fichiers metrics.json dans :

        results/<nom_modele>/metrics.json

    Returns
    -------
    dict
        {
            "mappo_gat_gat": contenu_metrics,
            "mappo_gcn_gcn": contenu_metrics,
            ...
        }
    """

    results_directory = Path(results_directory)

    if not results_directory.exists():
        raise FileNotFoundError(
            f"Le dossier n'existe pas : {results_directory}"
        )

    evaluation_logs: dict[str, dict[str, Any]] = {}

    for model_directory in sorted(results_directory.iterdir()):

        if not model_directory.is_dir():
            continue

        metrics_path = model_directory / "metrics.json"

        if not metrics_path.exists():
            print(
                f"[WARNING] metrics.json absent dans : "
                f"{model_directory}"
            )
            continue

        try:
            evaluation_logs[model_directory.name] = (
                load_json_file(metrics_path)
            )

            print(
                f"[INFO] Chargé : {metrics_path}"
            )

        except (ValueError, FileNotFoundError) as error:
            print(
                f"[WARNING] Impossible de charger "
                f"{metrics_path}: {error}"
            )

    if not evaluation_logs:
        raise ValueError(
            "Aucun fichier metrics.json valide trouvé "
            f"dans {results_directory}"
        )

    return evaluation_logs

def print_evaluation_statistics(
    evaluation_logs: dict,
    last_n_episodes: int = 10
) -> None:
    """
    Affiche les statistiques principales de chaque modèle
    dans le terminal.

    Parameters
    ----------
    evaluation_logs : dict
        Dictionnaire contenant les données JSON de chaque modèle.

    last_n_episodes : int
        Nombre de derniers épisodes utilisés pour calculer
        les performances finales.
    """

    print("\n" + "=" * 80)
    print("EVALUATION STATISTICS")
    print("=" * 80)

    for model_name, data in evaluation_logs.items():

        metrics = extract_episode_metrics(data)

        avg_rewards = metrics["avg_rewards"]
        std_rewards = metrics["std_rewards"]
        min_rewards = metrics["min_rewards"]
        max_rewards = metrics["max_rewards"]

        if avg_rewards.size == 0:
            print(f"\nModel: {model_name}")
            print("  Aucune donnée d'évaluation valide.")
            continue

        # Évite une erreur si le modèle contient moins
        # de last_n_episodes épisodes.
        number_of_final_episodes = min(
            last_n_episodes,
            avg_rewards.size
        )

        final_rewards = avg_rewards[
            -number_of_final_episodes:
        ]

        parameters = data.get("parameters", {})
        time_data = data.get("time", {})

        algorithm = parameters.get(
            "algorithm",
            "Unknown"
        )

        seed = parameters.get(
            "seed",
            "Unknown"
        )

        elapsed_time = time_data.get(
            "elapsed",
            "Unknown"
        )

        print(f"\nModel: {model_name}")
        print(f"  Algorithm             : {algorithm}")
        print(f"  Seed                  : {seed}")
        print(f"  Number of episodes    : {avg_rewards.size}")

        print(
            f"  Global average reward : "
            f"{np.mean(avg_rewards):.4f}"
        )

        print(
            f"  Global reward std     : "
            f"{np.std(avg_rewards):.4f}"
        )

        print(
            f"  Final average reward  : "
            f"{np.mean(final_rewards):.4f}"
        )

        print(
            f"  Final reward std      : "
            f"{np.std(final_rewards):.4f}"
        )

        print(
            f"  Best average reward   : "
            f"{np.max(avg_rewards):.4f}"
        )

        print(
            f"  Worst average reward  : "
            f"{np.min(avg_rewards):.4f}"
        )

        print(
            f"  Minimum reward        : "
            f"{np.min(min_rewards):.4f}"
        )

        print(
            f"  Maximum reward        : "
            f"{np.max(max_rewards):.4f}"
        )

        print(
            f"  Mean episode std      : "
            f"{np.mean(std_rewards):.4f}"
        )

        print(f"  Evaluation time       : {elapsed_time}")

    print("\n" + "=" * 80)

def generate_evaluation_plots(
    evaluation_directory: str | Path = "results",
    output_directory: str | Path = "plot_eval",
    moving_average_window: int = 5,
    last_n_episodes: int = 10
) -> None:

    evaluation_logs = load_evaluation_directory(
        evaluation_directory
    )

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"[INFO] {len(evaluation_logs)} modèles chargés."
    )

    print_evaluation_statistics(
        evaluation_logs,
        last_n_episodes=last_n_episodes
    )

    plot_evaluation_details(
        evaluation_logs=evaluation_logs,
        output_path=(
            output_directory
            / "evaluation_details.png"
        ),
        moving_average_window=moving_average_window
    )

    plot_average_reward_comparison(
        evaluation_logs=evaluation_logs,
        output_path=(
            output_directory
            / "evaluation_comparison.png"
        ),
        moving_average_window=moving_average_window,
        show_raw_rewards=True
    )

    plot_evaluation_summary(
        evaluation_logs=evaluation_logs,
        output_path=(
            output_directory
            / "evaluation_summary.png"
        ),
        last_n_episodes=last_n_episodes
    )

def main() -> None:
    """
    Charge les fichiers depuis results/
    et enregistre les figures dans plot_eval/.
    """

    results_directory = Path("results")
    output_directory = Path("plot_eval")

    generate_evaluation_plots(
        evaluation_directory=results_directory,
        output_directory=output_directory,
        moving_average_window=5,
        last_n_episodes=10
    )


if __name__ == "__main__":
    main()

