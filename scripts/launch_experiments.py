# scripts/launch_experiments.py

import argparse
import os
import subprocess
import sys




ALL_EXPERIMENTS = [
    {
        "name": "maddpg_gcn_gcn",
        "algorithm": "maddpg",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
        "resume": False,
    },
    {
        "name": "maddpg_gat_gat",
        "algorithm": "maddpg",
        "actor": "gnn_gat",
        "critic": "gnn_gat",
        "resume": False,
    },
    {
        "name": "ppo_gat_gat",
        "algorithm": "ppo",
        "actor": "gnn_gat",
        "critic": "gnn_gat",
        "resume": True,
    },   
    {
        "name": "mappo_gat_gat",
        "algorithm": "mappo",
        "actor": "gnn_gat",
        "critic": "gnn_gat",
        "resume": True,
    },
    
    {
        "name": "mappo_gcn_gcn",
        "algorithm": "mappo",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
        "resume": True,
    },
    {
        "name": "ppo_gcn_gcn",
        "algorithm": "ppo",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
        "resume": True,
    },
]

def build_command(
    exp,
    workers,
    max_threads,
    training_episodes,
    batch_size,
    epochs,
    resume = False,
):
    cmd = [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        "src.main",
        "train",

        "--override",
        "train.device=cpu",

        "--override",
        f"train.algorithm={exp['algorithm']}",

        "--override",
        f"actor.encoder={exp['actor']}",

        "--override",
        f"critic.encoder={exp['critic']}",

        "--override",
        f"run.name={exp['name']}",

        "--override",
        f"train.resume={resume}",

        "--override",
        f"train.workers={workers}",

        "--override",
        f"runtime.max_threads={max_threads}",
    ]

    if training_episodes is not None:
        cmd.extend(
            [
                "--override",
                f"train.training_episodes={training_episodes}",
            ]
        )

    if batch_size is not None:
        cmd.extend(
            [
                "--override",
                f"train.batch_size={batch_size}",
            ]
        )

    if epochs is not None:
        cmd.extend(
            [
                "--override",
                f"train.epochs={epochs}",
            ]
        )

    return cmd


def main(args):
    
    if args.num_trains < 1:
        raise ValueError("--num-trains must be >= 1")

    if args.num_trains > len(ALL_EXPERIMENTS):
        raise ValueError(
            f"--num-trains={args.num_trains}, but only "
            f"{len(ALL_EXPERIMENTS)} experiments are defined."
        )

    selected_experiments = ALL_EXPERIMENTS[: args.num_trains]

    processes = []


   
    for exp in selected_experiments:
        cmd = build_command(
            exp=exp,
            workers=args.workers,
            max_threads=args.max_threads,
            training_episodes=args.training_episodes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            resume=args.resume or exp.get("resume", False),
        )
        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
        )
        processes.append(
            (
                exp["name"],
                process,
            )
        )
    failed = False
    for name, process in processes:
        returncode = process.wait()
        if returncode == 0:
            print(
                f"✅ {name} finished successfully",
                flush=True,
            )
        else:
            failed = True
            print(
                f"❌ {name} failed with returncode {returncode}",
                flush=True,
            )
    if failed:
        raise SystemExit(1)
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch multiple PPO experiments in parallel"
    )

    parser.add_argument(
        "--num-trains",
        type=int,
        default=3,
        help="Number of experiments to launch",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=25,
        help="Workers per training",
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=30,
        help="Max threads per training",
    )

    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Whether to resume training",
    )

    parser.add_argument(
        "--training-episodes",
        type=int,
        default=None,
        help="Override train.training_episodes",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override train.batch_size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs",
    )

    parser.add_argument(
        "--tpu-monitor",
        action="store_true",
        help="If XLA/TPU is available, start one TPU monitor in the launcher process",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between launching trainings",
    )

    args = parser.parse_args()

    main(args)