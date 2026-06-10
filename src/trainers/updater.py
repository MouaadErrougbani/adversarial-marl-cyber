# src/trainers/updater.py

from joblib import Parallel, delayed


def train_models(agents):
    device_str = str(agents[0].device).lower()

    if "cpu" not in device_str:
        losses = []

        for agent in agents:
            losses.append(agent.learn())

        return losses

    agent_count = len(agents)

    def learn(i):
        return agents[i].learn()

    return Parallel(
        prefer="threads",
        n_jobs=agent_count,
    )(
        delayed(learn)(i)
        for i in range(agent_count)
    )