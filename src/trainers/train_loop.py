# src/trainers/train_loop.py
import copy
import time

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
    cfg,
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

    device_str = str(device).lower()

    if "cpu" in device_str:
        rollout_agents = agents
    else:
        rollout_agents = [
            copy.deepcopy(agent).to("cpu")
            for agent in agents
        ]

    for agent in rollout_agents:
        agent.train()

    agent_count = len(agents)

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
            f"Episode {start_ep} -> {end_ep}",
            "=" * 20,
            flush=True,
        )

        #
        # Rollout collection
        #

        
        rollout_data = collect_data(
            rollout_agents,
            envs,
            hp,
            agent_count,
            max_threads,
        )

        memories, rewards = zip(
            *rollout_data
        )

        memories = [
            list(m)
            for m in zip(*memories)
        ]

        for i in range(agent_count):

            agents[i].memory.mems = (
                memories[i]
            )

        #
        # PPO update
        #

        losses = train_models(
            agents
        )

        for rollout_agent, train_agent in zip(rollout_agents, agents):
            actor_state = {
                k: v.detach().cpu()
                for k, v in train_agent.actor.state_dict().items()
            }

            critic_state = {
                k: v.detach().cpu()
                for k, v in train_agent.critic.state_dict().items()
            }

            rollout_agent.actor.load_state_dict(actor_state)
            rollout_agent.critic.load_state_dict(critic_state)
            rollout_agent.to("cpu")
            rollout_agent.train()

        losses_str = ",".join(
            [
                f"{loss:.4f}"
                for loss in losses
            ]
        )

        print(
            f"[{update}] "
            f"Loss: [{losses_str}]",
            flush=True,
        )

        avg_reward = (
            sum(rewards)
            / len(rewards)
        )

        avg_loss = (
            sum(losses)
            / len(losses)
        )

        print(
            f"Avg reward: "
            f"{avg_reward:.4f}",
            flush=True,
        )

        #
        # Logging
        #

        log.append(
            (
                avg_reward,
                update,
                avg_loss,
            )
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