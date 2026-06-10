# src/trainers/config.py

from types import SimpleNamespace


def default_config():
    return {
        "paths": {
            "logs": "logs",
            "checkpoints": "checkpoints",
        },
        "runtime": {
            "max_threads": 36,
            "max_training_hours": 11.1,
        },
        "train": {
            "seed": 1337,
            "episode_len": 500,
            "episodes_per_update": 25,
            "workers": 25,
            "batch_size": 2500,
            "training_episodes": 50_000,
            "epochs": 4,
            "resume": False,
            "resume_name": None,
            "resume_start_iter": None,
        },
        "model": {
            "hidden": 256,
            "embedding": 128,
            "actor_lr": 0.0003,
            "critic_lr": 0.001,
            "clip": 0.2,
        },
        "run": {
            "name": "model",
        },
    }


def build_hyper_params(cfg):
    return SimpleNamespace(
        N=cfg["train"]["episodes_per_update"],
        workers=cfg["train"]["workers"],
        bs=cfg["train"]["batch_size"],
        episode_len=cfg["train"]["episode_len"],
        training_episodes=cfg["train"]["training_episodes"],
        epochs=cfg["train"]["epochs"],
        fnames=cfg["run"]["name"],
    )

