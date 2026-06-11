# scripts/launch_experiments.py

import argparse
import subprocess
import sys
import time


ALL_EXPERIMENTS = [
    {
        "name": "ppo_gcn_gcn",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
    },
    {
        "name": "ppo_gat_gat",
        "actor": "gnn_gat",
        "critic": "gnn_gat",
    },
    {
        "name": "ppo_gat_gcn",
        "actor": "gnn_gat",
        "critic": "gnn_gcn",
    },
    {
        "name": "ppo_gcn_gat",
        "actor": "gnn_gcn",
        "critic": "gnn_gat",
    },
]


def build_command(
    exp,
    device,
    workers,
    max_threads,
    training_episodes,
    batch_size,
    epochs,
    tpu_monitor,
):
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "train",

        "--override",
        f"train.device={device}",

        "--override",
        f"actor.encoder={exp['actor']}",

        "--override",
        f"critic.encoder={exp['critic']}",

        "--override",
        f"run.name={exp['name']}",

        "--override",
        f"train.workers={workers}",

        "--override",
        f"runtime.max_threads={max_threads}",

        "--override",
        f"runtime.tpu_monitor={str(tpu_monitor).lower()}",
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


def main():
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for each training",
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
        help="Enable TPU monitor",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between launching trainings",
    )

    args = parser.parse_args()

    if args.num_trains > len(ALL_EXPERIMENTS):
        raise ValueError(
            f"num-trains={args.num_trains} but only "
            f"{len(ALL_EXPERIMENTS)} experiments are defined."
        )

    selected_experiments = ALL_EXPERIMENTS[: args.num_trains]

    total_workers = args.num_trains * args.workers
    total_threads = args.num_trains * args.max_threads

    print("\n=== PARALLEL TRAINING PLAN ===")
    print(f"Number of trainings: {args.num_trains}")
    print(f"Workers per training: {args.workers}")
    print(f"Max threads per training: {args.max_threads}")
    print(f"Estimated total workers: {total_workers}")
    print(f"Estimated total max threads: {total_threads}")
    print(f"Device: {args.device}")
    print("==============================\n")

    processes = []

    for exp in selected_experiments:
        cmd = build_command(
            exp=exp,
            device=args.device,
            workers=args.workers,
            max_threads=args.max_threads,
            training_episodes=args.training_episodes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            tpu_monitor=args.tpu_monitor,
        )

        print(
            f"Starting {exp['name']} "
            f"actor={exp['actor']} critic={exp['critic']}",
            flush=True,
        )

        process = subprocess.Popen(cmd)
        processes.append(
            (
                exp["name"],
                process,
            )
        )

        time.sleep(args.delay)

    print("\nAll trainings launched.\n", flush=True)

    for name, process in processes:
        returncode = process.wait()

        print(
            f"{name} finished with returncode {returncode}",
            flush=True,
        )


if __name__ == "__main__":
    main()