# src/trainers/updater.py

import torch
from joblib import Parallel, delayed


def train_models(agents, max_threads, num_agents):


    def learn(i):
        if i < 4:
                torch.set_num_threads(max_threads // 9)
        else:
                torch.set_num_threads((max_threads // 9) * num_agents)
        return agents[i].learn()

    return Parallel(
        prefer="threads",
        n_jobs=num_agents,
    )(
        delayed(learn)(i)
        for i in range(num_agents)
    )