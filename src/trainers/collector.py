# src/trainers/collector.py


from joblib import Parallel, delayed
from src import MultiPPOMemory
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_episode(agents, env, hp, agent_count, max_threads, i):
    torch.set_num_threads(max_threads // hp.workers)

    env.reset()
    states = env.last_obs
    blocked_rewards = [0] * agent_count

    tot_reward = 0
    memory_buffers = MultiPPOMemory(hp.bs, agents=agent_count)

    for ts in tqdm(range(hp.episode_len), desc=f"Generating episode {i}"):
        actions = dict()
        memories = dict()

        for k, (state, blocked) in states.items():
            i = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[i].get_action((state, blocked))
                memories[i] = (state, action, value, prob)
                actions[k] = action

        next_state, rewards, _, _, _ = env.step(actions)
        rewards = list(rewards.values())
        tot_reward += sum(rewards) / agent_count

        for i in range(agent_count):
            if i in memories:
                s, a, v, p = memories[i]
                r = rewards[i] + blocked_rewards[i]
                t = 0 if ts < hp.episode_len - 1 else 1

                memory_buffers.remember(i, s, a, v, p, r, t)
                blocked_rewards[i] = 0
            else:
                blocked_rewards[i] += rewards[i]

        states = next_state

    return memory_buffers.mems, tot_reward

def collect_data(
    agents,
    envs,
    hp,
    agent_count,
    max_threads,
):

    out = Parallel(prefer="processes", n_jobs=hp.workers)(
            delayed(generate_episode)(agents, envs[i % len(envs)], hp, agent_count, max_threads, i)
            for i in range(hp.N)
        )

    return out
