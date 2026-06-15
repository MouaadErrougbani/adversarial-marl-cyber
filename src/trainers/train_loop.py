# src/trainers/train_loop.py

import copy
import time
import numpy as np

from .collector import (
    collect_data,
)

from .updater import (
    train_models,
)

from .checkpoint import (
    save_logs,
    save_checkpoints,
)


def train_loop(
    agents,
    envs,
    hp,
    fnames,
    log,
    start_iter,
    log_dir,
    checkpoint_dir,
    max_threads,
    device="cpu",
):
    """
    Main PPO training loop.
    """
    for agent in agents:
        agent.train()
    
    

    num_agents = len(agents)

    total_updates = (
        hp.training_episodes
        // hp.N
    )

    for update in range(
        start_iter,
        total_updates,
    ):

        start_ep = update * hp.N
        end_ep = (
            update + 1
        ) * hp.N

        print(
            "=" * 20,
            f"Episode {start_ep} --{fnames}--> {end_ep}",
            "=" * 20,
            flush=True,
        )

        #
        # Rollout collection
        #

        collection_start = time.perf_counter()

        rollout_data = collect_data(
            agents,
            envs,
            hp,
            num_agents,
            max_threads,
        )

        collection_end = time.perf_counter()
        collection_time_sec = collection_end - collection_start

        memories, rewards, agent_rewards = zip(
            *rollout_data
        )

        memories = [
            list(m)
            for m in zip(*memories)
        ]

        for i in range(num_agents):
            agents[i].memory.memories = (
                memories[i]
            )

        #
        # PPO update
        #

        training_start = time.perf_counter()
       
        losses = train_models(
            agents
        )

        training_end = time.perf_counter()
        training_time_sec = training_end - training_start

        #
        # Sync rollout agents only if training is not CPU
        #

        #
        # Metrics: rewards
        #

        rewards_np = np.array(
            rewards,
            dtype=np.float32,
        )

        avg_reward = float(
            rewards_np.mean()
        )

        std_reward = float(
            rewards_np.std()
        )

        min_reward = float(
            rewards_np.min()
        )

        max_reward = float(
            rewards_np.max()
        )

        agent_rewards_np = np.array(
            agent_rewards,
            dtype=np.float32,
        )

        avg_rewards_agents = agent_rewards_np.mean(
            axis=0
        )

        #
        # Metrics: losses
        #

        total_losses = [
            float(loss["total_loss"])
            for loss in losses
        ]

        actor_losses = [
            float(loss["actor_loss"])
            for loss in losses
        ]

        critic_losses = [
            float(loss["critic_loss"])
            for loss in losses
        ]

        total_losses_np = np.array(
            total_losses,
            dtype=np.float32,
        )

        avg_loss = float(
            total_losses_np.mean()
        )

        min_loss = float(
            total_losses_np.min()
        )

        max_loss = float(
            total_losses_np.max()
        )

        #
        # Print
        #




        print(
            f"Avg reward: {avg_reward:.4f} | "
            f"Avg loss: {avg_loss:.4f} | "
            f"Collect: {collection_time_sec:.2f}s | "
            f"Train: {training_time_sec:.2f}s",
            flush=True,
        )

        #
        # Logging
        #

        log_entry = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,

            "avg_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,

            "collection_time_sec": collection_time_sec,
            "training_time_sec": training_time_sec,
        }

        for i, reward in enumerate(avg_rewards_agents):
            log_entry[f"avg_reward_agent_{i}"] = float(
                reward
            )

        for i in range(num_agents):
            log_entry[f"total_loss_agent_{i}"] = float(
                total_losses[i]
            )

            log_entry[f"actor_loss_agent_{i}"] = float(
                actor_losses[i]
            )

            log_entry[f"critic_loss_agent_{i}"] = float(
                critic_losses[i]
            )

        log.append(
            log_entry
        )

        save_logs(
            log=log,
            log_dir=log_dir,
            run_name=hp.fnames,
        )

        #
        # Checkpoints
        #

        save_checkpoints(
            agents,
            checkpoint_dir,
            checkpoint_name=hp.fnames,
            update_idx=update,
            episodes_per_update=hp.N,
        )

    print(
        "\nTraining finished.",
        flush=True,
    )