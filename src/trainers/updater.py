# src/trainers/updater.py

from joblib import Parallel, delayed


def train_models(agents):

    num_agents = len(agents)

    def learn(i):
        import time

        t0 = time.perf_counter()
        result = agents[i].learn()
        t1 = time.perf_counter()

        print(
            f"Agent {i}: {t1-t0:.2f}s",
            flush=True
        )

        return result

    return Parallel(
        prefer="threads",
        n_jobs=num_agents,
    )(
        delayed(learn)(i)
        for i in range(num_agents)
    )