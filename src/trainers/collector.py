# src/trainers/collector.py

from joblib import Parallel, delayed
from src import MultiPPOMemory
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_episode(
    agents,
    env,
    hp,
    agent_count,
    max_threads,
    episode_idx,
):
    threads_per_worker = max(
        1,
        max_threads // hp.workers,
    )

    torch.set_num_threads(
        threads_per_worker
    )

    env.reset()
    states = env.last_obs

    blocked_rewards = [0.0] * agent_count
    agent_total_rewards = [0.0] * agent_count

    tot_reward = 0.0

    memory_buffers = MultiPPOMemory(
        hp.bs,
        agents=agent_count,
    )

    for ts in tqdm(
        range(hp.episode_len),
        desc=f"Generating episode {episode_idx}",
    ):
        actions = dict()
        memories = dict()

        for k, (state, blocked) in states.items():
            agent_idx = int(k[-1])

            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[agent_idx].get_action(
                    (
                        state,
                        blocked,
                    )
                )

                memories[agent_idx] = (
                    state,
                    action,
                    value,
                    prob,
                )

                actions[k] = action

        next_state, rewards, _, _, _ = env.step(
            actions
        )

        rewards = list(
            rewards.values()
        )

        for agent_idx in range(agent_count):
            agent_total_rewards[agent_idx] += rewards[agent_idx]

        tot_reward += (
            sum(rewards)
            / agent_count
        )

        for agent_idx in range(agent_count):
            if agent_idx in memories:
                s, a, v, p = memories[agent_idx]

                r = (
                    rewards[agent_idx]
                    + blocked_rewards[agent_idx]
                )

                t = (
                    0
                    if ts < hp.episode_len - 1
                    else 1
                )

                memory_buffers.remember(
                    agent_idx,
                    s,
                    a,
                    v,
                    p,
                    r,
                    t,
                )

                blocked_rewards[agent_idx] = 0.0

            else:
                blocked_rewards[agent_idx] += rewards[agent_idx]

        states = next_state

    return (
        memory_buffers.mems,
        tot_reward,
        agent_total_rewards,
    )


def collect_data(
    agents,
    envs,
    hp,
    agent_count,
    max_threads,
):
    out = Parallel(
        prefer="processes",
        n_jobs=hp.workers,
    )(
        delayed(generate_episode)(
            agents,
            envs[i % len(envs)],
            hp,
            agent_count,
            max_threads,
            i,
        )
        for i in range(hp.N)
    )

    return out