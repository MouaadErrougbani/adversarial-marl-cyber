# src/trainers/updater.py

from joblib import Parallel, delayed


def train_models(agents):

    num_agents = len(agents)

    def learn(i):
        return agents[i].learn()

    return Parallel(
        prefer="threads",
        n_jobs=num_agents,
    )(
        delayed(learn)(i)
        for i in range(num_agents)
    )