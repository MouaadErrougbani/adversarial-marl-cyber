import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_reward_with_std_band(logs: dict, save_path: str):
    """
    Trace le reward moyen avec la bande de déviation standard.
    Permet de visualiser simultanément performance et stabilité.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        avg_rewards = np.array([
            step["avg_reward"]
            for step in history
        ])

        std_rewards = np.array([
            step["std_reward"]
            for step in history
        ])

        iterations = np.arange(len(avg_rewards))

        lower = avg_rewards - std_rewards
        upper = avg_rewards + std_rewards

        # Courbe moyenne
        plt.plot(
            iterations,
            avg_rewards,
            linewidth=2,
            label=model_name,
        )

        # Bande de variance
        plt.fill_between(
            iterations,
            lower,
            upper,
            alpha=0.2,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward with Standard Deviation Band")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_total_loss_per_agent(
    logs: dict,
    path: str
):
    """
    Trace l'évolution de la loss totale.

    Pour MAPPO :
    - trace uniquement total_loss_agent_0 ;
    - n'affiche pas la légende de l'agent ;
    - considère cette courbe comme la loss du critic centralisé.

    Pour les autres modèles :
    - trace les losses des 5 agents.
    """

    n_models = len(logs)

    if n_models == 0:
        return

    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, (model_name, history) in zip(axes, logs.items()):

        is_mappo = model_name.lower().startswith("mappo")

        # MAPPO utilise la loss stockée dans agent 0
        agent_ids = [0] if is_mappo else range(5)

        curves_plotted = 0
        all_losses = []

        for agent_id in agent_ids:

            key = f"total_loss_agent_{agent_id}"

            losses = [
                step[key]
                for step in history
                if key in step and step[key] is not None
            ]

            if not losses:
                continue

            iterations = range(len(losses))

            ax.plot(
                iterations,
                losses,
                marker="o",
                linewidth=2,
                label=None if is_mappo else f"Agent {agent_id}"
            )

            all_losses.extend(losses)
            curves_plotted += 1

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Loss")
        ax.grid(True, linestyle="--", alpha=0.5)

        if is_mappo:
            ax.set_title(
                f"Total Loss - {model_name}\nCentralized Critic"
            )
        else:
            ax.set_title(
                f"Total Loss per Agent - {model_name}"
            )

        if not is_mappo and curves_plotted > 0:
            ax.legend()

        if all_losses:
            if all(value > 0 for value in all_losses):
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=1e-3)
        else:
            ax.text(
                0.5,
                0.5,
                "Aucune loss disponible",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

    for ax in axes[n_models:]:
        ax.set_visible(False)

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.suptitle(
        "Total Loss — Comparaison des algorithmes",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def plot_collection_time(logs: dict, save_path: str):
    """
    Trace le temps de collecte des expériences.
    Permet d'évaluer le coût des interactions avec l'environnement.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        collection_times = [
            step["collection_time_sec"]
            for step in history
            if "collection_time_sec" in step
        ]

        iterations = range(len(collection_times))

        plt.plot(
            iterations,
            collection_times,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Collection Time (sec)")
    plt.title("Collection Time vs Iteration")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_actor_loss_per_agent(
    logs: dict,
    path: str
):
    """
    Trace l'évolution de l'actor loss pour chaque agent.

    Une seule figure est enregistrée, avec un sous-graphe
    pour chaque modèle/algorithme.
    """

    n_models = len(logs)

    # Organisation automatique des sous-graphiques
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 5 * n_rows),
        squeeze=False
    )

    # Transforme la matrice d'axes en liste
    axes = axes.flatten()

    for ax, (model_name, history) in zip(axes, logs.items()):

        for agent_id in range(5):
            key = f"actor_loss_agent_{agent_id}"

            losses = [
                step[key]
                for step in history
                if key in step
            ]

            if not losses:
                continue

            iterations = range(len(losses))

            ax.plot(
                iterations,
                losses,
                marker="o",
                linewidth=2,
                label=f"Agent {agent_id}"
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Actor Loss")
        ax.set_title(f"Actor Loss per Agent - {model_name}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    # Masquer les sous-graphiques inutilisés
    for ax in axes[n_models:]:
        ax.set_visible(False)

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.suptitle(
        "Actor Loss per Agent — Comparaison des algorithmes",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_critic_loss(logs: dict, save_path: str):
    """
    Trace l'évolution de la critic loss.
    Permet d'analyser la convergence du critic.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        critic_losses = [
            step["critic_loss_agent_0"]
            for step in history
            if "critic_loss_agent_0" in step
        ]

        iterations = range(len(critic_losses))

        plt.plot(
            iterations,
            critic_losses,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss vs Iteration")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reward_vs_training_time(logs: dict, save_path: str):
    """
    Trace la relation entre reward et temps de training.
    Permet d'évaluer l'efficacité des modèles.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        rewards = [
            step["avg_reward"]
            for step in history
        ]

        training_times = [
            step["training_time_sec"]
            for step in history
        ]

        plt.scatter(
            training_times,
            rewards,
            s=50,
            label=model_name,
        )

    plt.xlabel("Training Time (sec)")
    plt.ylabel("Average Reward")
    plt.title("Reward vs Training Time")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_time(logs: dict, save_path: str):
    """
    Trace le temps de training par itération.
    Permet de comparer le coût computationnel des modèles.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        training_times = [
            step["training_time_sec"]
            for step in history
            if "training_time_sec" in step
        ]

        iterations = range(len(training_times))

        plt.plot(
            iterations,
            training_times,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Training Time (sec)")
    plt.title("Training Time per Iteration")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reward_range(logs: dict, save_path: str):
    """
    Trace la plage des rewards entre min_reward et max_reward.
    Permet de visualiser la dispersion des performances.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        avg_rewards = [
            step["avg_reward"]
            for step in history
        ]

        min_rewards = [
            step["min_reward"]
            for step in history
        ]

        max_rewards = [
            step["max_reward"]
            for step in history
        ]

        iterations = range(len(avg_rewards))

        # Courbe moyenne
        plt.plot(
            iterations,
            avg_rewards,
            linewidth=2,
            label=model_name
        )

        # Zone min-max
        plt.fill_between(
            iterations,
            min_rewards,
            max_rewards,
            alpha=0.2
        )

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Reward Range During Training")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_std_reward(logs: dict, save_path: str):
    """
    Trace l'évolution du std_reward pour chaque modèle.
    Permet d'évaluer la stabilité de l'apprentissage.
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        std_rewards = [
            step["std_reward"]
            for step in history
            if "std_reward" in step
        ]

        iterations = range(len(std_rewards))

        plt.plot(
            iterations,
            std_rewards,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Std Reward")
    plt.title("Reward Stability Across Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_avg_reward(logs: dict, save_path: str):
    """
    Plot avg_reward vs iteration pour tous les modèles.

    Parameters
    ----------
    logs : dict
        {
            "mappo_gat_gat": [dict, dict, ...],
            "ppo_gat_gat": [dict, dict, ...],
        }

    save_path : str
        Exemple:
        "plots/avg_reward.png"
    """

    plt.figure(figsize=(10, 6))

    for model_name, history in logs.items():

        rewards = [
            step["avg_reward"]
            for step in history
            if "avg_reward" in step
        ]

        iterations = range(len(rewards))

        plt.plot(
            iterations,
            rewards,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Iteration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def load_logs(log_path: str) -> dict:
    """
    Charge tous les fichiers .pt dans log_path.

    Retour:
    {
        "mappo_gat_gat": x,
        "ppo_gat_gat": x,
        ...
    }
    """

    logs = {}

    for filename in os.listdir(log_path):
        if not filename.endswith(".pt"):
            continue

        model_name = filename.replace(".pt", "")
        file_path = os.path.join(log_path, filename)

        x = torch.load(file_path, map_location="cpu")
        logs[model_name] = x

    return logs

def generate_all_plots(logs: dict, output_dir: str = "plots"):
    """
    Génère tous les plots disponibles.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Plots globaux
    plot_avg_reward(
        logs,
        os.path.join(output_dir, "avg_reward.png")
    )

    plot_std_reward(
        logs,
        os.path.join(output_dir, "std_reward.png")
    )

    plot_reward_range(
        logs,
        os.path.join(output_dir, "reward_range.png")
    )

    plot_reward_with_std_band(
        logs,
        os.path.join(output_dir, "reward_std_band.png")
    )

    plot_training_time(
        logs,
        os.path.join(output_dir, "training_time.png")
    )

    plot_collection_time(
        logs,
        os.path.join(output_dir, "collection_time.png")
    )

    plot_reward_vs_training_time(
        logs,
        os.path.join(output_dir, "reward_vs_training_time.png")
    )

    plot_critic_loss(
        logs,
        os.path.join(output_dir, "critic_loss.png")
    )

    # Plots par modèle
    plot_actor_loss_per_agent(
        logs,
        os.path.join(output_dir, "actor_loss.png")
    )


    plot_total_loss_per_agent(
        logs,
        os.path.join(output_dir, "total_loss.png")
    )

    print(f"All plots saved in: {output_dir}")




if __name__ == "__main__":

    logs = load_logs("logs")

    print("Loaded models:")
    for model_name, data in logs.items():
        print(
            f"  - {model_name}: "
            f"{len(data)} iterations"
        )

    generate_all_plots(
        logs,
        output_dir="plots"
    )