# src/trainers/collector.py

from joblib import Parallel, delayed
import torch



def generate_episode_ppo(
    agents,
    env,
    hp,
    num_agents,
    max_threads,
    episode_idx,
    ):
    from src import MultiPPOMemory
    threads_per_worker = max(
        1,
        max_threads // hp.workers,
    )

    torch.set_num_threads(
        threads_per_worker
    )

    env.reset()
    states = env.last_obs

    blocked_rewards = [0.0] * num_agents
    agent_total_rewards = [0.0] * num_agents

    tot_reward = 0.0

    memory_buffers = MultiPPOMemory(
        batch_size=hp.bs,
        num_agents=num_agents,
    )

    for ts in range(hp.episode_len):
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

        for agent_idx in range(num_agents):
            agent_total_rewards[agent_idx] += rewards[agent_idx]

        tot_reward += (
            sum(rewards)
            / num_agents
        )

        for agent_idx in range(num_agents):
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
        memory_buffers.memories,
        tot_reward,
        agent_total_rewards,
    )

def generate_episode_mappo(
    agents,
    env,
    hp,
    num_agents,
    max_threads,
    episode_idx,
    ):
    from src import MultiMAPPOMemory

    threads_per_worker = max(
        1,
        max_threads // hp.workers,
    )

    torch.set_num_threads(
        threads_per_worker
    )

    env.reset()
    states = env.last_obs

    blocked_rewards = [0.0] * num_agents
    agent_total_rewards = [0.0] * num_agents

    tot_reward = 0.0

    memory_buffers = MultiMAPPOMemory(
        batch_size=hp.bs,
        num_agents=num_agents,
    )

    for ts in range(hp.episode_len):
        actions = dict()
        memories = dict()
        global_observation  = [
            states[f"blue_agent_{i}"][0]
            for i in range(num_agents)
        ]
        for k, (state, blocked) in states.items():
            agent_idx = int(k[-1])

            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[agent_idx].get_action(
                    (
                        state,
                        global_observation,
                        blocked,
                    )
                )

                memories[agent_idx] = (
                    state,
                    global_observation,
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

        for agent_idx in range(num_agents):
            agent_total_rewards[agent_idx] += rewards[agent_idx]

        tot_reward += (
            sum(rewards)
            / num_agents
        )
        for agent_idx in range(num_agents):
            if agent_idx in memories:
                s, g, a, v, p = memories[agent_idx]

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
                    g,
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
        memory_buffers.memories,
        tot_reward,
        agent_total_rewards,
    )



episode_generator_registry  = {
    "PPO": generate_episode_ppo,
    "MAPPO": generate_episode_mappo
}


@torch.no_grad()
def generate_episode(
    agents,
    env,
    hp,
    num_agents,
    max_threads,
    episode_idx,
    ):
    algorithm = agents[0].algorithm.upper()
    if algorithm not in episode_generator_registry :
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    generate_episode_fn = episode_generator_registry[algorithm]
    return generate_episode_fn(
        agents,
        env,
        hp,
        num_agents,
        max_threads,
        episode_idx,
    )

def collect_data(
    agents,
    envs,
    hp,
    num_agents,
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
            num_agents,
            max_threads,
            i,
        )
        for i in range(hp.N)
    )

    return out