# src/trainers/trainer.py

import os
import torch

from .config import (
    build_hyper_params,
)

from .builders import (
    build_agents,
    build_envs,
)

from .checkpoint import (
    load_logs,
    load_checkpoints,
)

from .train_loop import (
    train_loop,
)




def run_train(cfg, device = "cpu"):
    """
    Main training entrypoint.
    """

    seed = cfg["train"]["seed"]


    max_threads = cfg["runtime"]["max_threads"]


    torch.manual_seed(seed)
    torch.set_num_threads(max_threads)

    hp = build_hyper_params(cfg)

    log_dir = cfg["paths"]["logs"]
    checkpoint_dir = cfg["paths"]["checkpoints"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    #
    # Logs / Resume
    #

    resume_name = cfg["train"].get("resume_name") or cfg["run"]["name"]

    log, start_iter = load_logs(
        log_dir=log_dir,
        resume=cfg["train"].get(
            "resume",
            False,
        ),
        resume_name=resume_name,
        override_start_iter=cfg[
            "train"
        ].get(
            "resume_start_iter"
        ),
    )

    #
    # Agents
    #

    agents = build_agents(
        cfg,
    )

    #
    # Checkpoints
    #

    if cfg["train"].get(
        "resume",
        False,
    ):

        load_checkpoints(
            agents=agents,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=resume_name,
        )

    #
    # Environments
    #

    envs = build_envs(
        cfg
    )

    #
    # Print config
    #

    print("\n" + "=" * 80, flush=True)

    print("TRAINING CONFIG", flush=True)

    print("=" * 80, flush=True)

    print(
        f"Seed: {seed}", flush=True
    )

    print(
        f"Workers: {hp.workers}", flush=True
    )

    print(
        f"Episode length: "
        f"{hp.episode_len}", flush=True
    )

    print(
        f"Episodes per update: "
        f"{hp.N}", flush=True
    )

    print(
        f"Training episodes: "
        f"{hp.training_episodes}", flush=True
    )

    print(
        f"Batch size: "
        f"{hp.bs}", flush=True
    )

    print(
        f"Epochs: "
        f"{hp.epochs}", flush=True
    )

    print(
        f"Run name: "
        f"{hp.fnames}", flush=True
    )

    print(
        f"Training device: {device}", flush=True
    )

    print("=" * 80, flush=True)

    #
    # Main Loop
    #

    train_loop(
        agents=agents,
        envs=envs,
        hp=hp,
        fnames=hp.fnames,
        log=log,
        start_iter=start_iter,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_threads=max_threads,
    )