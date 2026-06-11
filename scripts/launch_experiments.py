# scripts/launch_experiments.py

import argparse
import os
import subprocess
import sys
import time

from src.trainers.tpu import (
    start_tpu_test_async,
    stop_tpu_test_async,
)


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
):
    cmd = [
        sys.executable,
        "-W",
        "ignore",
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
        "train.resume=false",

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
        choices=["cpu", "cuda", "auto", "xla"],
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
        help="Start one TPU monitor in the launcher process",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between launching trainings",
    )

    args = parser.parse_args()

    if args.num_trains < 1:
        raise ValueError("--num-trains must be >= 1")

    if args.num_trains > len(ALL_EXPERIMENTS):
        raise ValueError(
            f"--num-trains={args.num_trains}, but only "
            f"{len(ALL_EXPERIMENTS)} experiments are defined."
        )

    selected_experiments = ALL_EXPERIMENTS[: args.num_trains]

    total_workers = args.num_trains * args.workers
    total_threads = args.num_trains * args.max_threads

    print("\n=== PARALLEL TRAINING PLAN ===", flush=True)
    print(f"Number of trainings: {args.num_trains}", flush=True)
    print(f"Workers per training: {args.workers}", flush=True)
    print(f"Max threads per training: {args.max_threads}", flush=True)
    print(f"Estimated total workers: {total_workers}", flush=True)
    print(f"Estimated total max threads: {total_threads}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"TPU monitor: {args.tpu_monitor}", flush=True)
    print("==============================\n", flush=True)

    processes = []
    tpu_monitor_started = False

    if args.tpu_monitor and args.device != "xla":
        tpu_monitor_started = start_tpu_test_async(
            interval_seconds=30 * 60,
            batch=64,
            seq_len=512,
            hidden=1024,
            layers=12,
            heads=16,
            steps=100,
        )

        if tpu_monitor_started:
            print("[launcher] TPU monitor started", flush=True)
        else:
            print("[launcher] TPU monitor already running", flush=True)

    try:
        for exp in selected_experiments:
            cmd = build_command(
                exp=exp,
                device=args.device,
                workers=args.workers,
                max_threads=args.max_threads,
                training_episodes=args.training_episodes,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )

            print(
                f"Starting {exp['name']} "
                f"actor={exp['actor']} "
                f"critic={exp['critic']}",
                flush=True,
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

            time.sleep(args.delay)

        print("\nAll trainings launched.\n", flush=True)

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

    finally:
        if tpu_monitor_started:
            stop_tpu_test_async()
            print("[launcher] TPU monitor stopped", flush=True)


if __name__ == "__main__":
    main()