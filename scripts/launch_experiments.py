# scripts/launch_experiments.py

import argparse
import os
import subprocess
import sys
import time

# from src.trainers.tpu import (
#     start_tpu_test_async,
#     stop_tpu_test_async,
# )

def start_tpu_test_async(
    batch=64,
    seq_len=512,
    hidden=1024,
    layers=12,
    heads=16,
    steps=100,
):
    global _TPU_TEST_PROCESS

    if _TPU_TEST_PROCESS is not None and _TPU_TEST_PROCESS.poll() is None:
        return False

    env = os.environ.copy()
    env["PJRT_DEVICE"] = "TPU"

    _TPU_TEST_PROCESS = subprocess.Popen(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "src.utils.tpu_test_runner",
            "--batch",
            str(batch),
            "--seq-len",
            str(seq_len),
            "--hidden",
            str(hidden),
            "--layers",
            str(layers),
            "--heads",
            str(heads),
            "--steps",
            str(steps),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return True

def check_tpu_test_async():
    """
    Vérifie si le TPU test est terminé.
    Ne bloque pas le training.
    """

    global _TPU_TEST_PROCESS

    if _TPU_TEST_PROCESS is None:
        return None

    # مازال خدام
    if _TPU_TEST_PROCESS.poll() is None:
        return None

    stdout, stderr = _TPU_TEST_PROCESS.communicate()
    returncode = _TPU_TEST_PROCESS.returncode

    _TPU_TEST_PROCESS = None

  
    if returncode == 0:
        return True

    if stderr and "Device or resource busy" in stderr:
        return False

    return False



ALL_EXPERIMENTS = [
    {
        "name": "ppo_gcn_gcn",
        "algorithm": "ppo",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
    },
    {
        "name": "mappo_gcn_gcn",
        "algorithm": "mappo",
        "actor": "gnn_gcn",
        "critic": "gnn_gcn",
    },
    {
        "name": "ppo_gat_gat",
        "algorithm": "ppo",
        "actor": "gnn_gat",
        "critic": "gnn_gat",
    },
    {
        "name": "ppo_gat_gcn",
        "algorithm": "ppo",
        "actor": "gnn_gat",
        "critic": "gnn_gcn",
    },
    {
        "name": "ppo_gcn_gat",
        "algorithm": "ppo",
        "actor": "gnn_gcn",
        "critic": "gnn_gat",
    },
]

def build_command(
    exp,
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

    if args.num_trains < 1:
        raise ValueError("--num-trains must be >= 1")

    if args.num_trains > len(ALL_EXPERIMENTS):
        raise ValueError(
            f"--num-trains={args.num_trains}, but only "
            f"{len(ALL_EXPERIMENTS)} experiments are defined."
        )

    selected_experiments = ALL_EXPERIMENTS[: args.num_trains]

    processes = []
    tpu_monitor_started = False

    check_tpu_test_async()
    start_tpu_test_async(
            batch=64,
            seq_len=512,
            hidden=1024,
            layers=12,
            heads=16,
            steps=100,
        )


    # try:
    #     for exp in selected_experiments:
    #         cmd = build_command(
    #             exp=exp,
    #             workers=args.workers,
    #             max_threads=args.max_threads,
    #             training_episodes=args.training_episodes,
    #             batch_size=args.batch_size,
    #             epochs=args.epochs,
    #         )


    #         process = subprocess.Popen(
    #             cmd,
    #             env=os.environ.copy(),
    #         )

    #         processes.append(
    #             (
    #                 exp["name"],
    #                 process,
    #             )
    #         )

    #         time.sleep(args.delay)


    #     failed = False

    #     for name, process in processes:
    #         returncode = process.wait()

    #         if returncode == 0:
    #             print(
    #                 f"✅ {name} finished successfully",
    #                 flush=True,
    #             )
    #         else:
    #             failed = True
    #             print(
    #                 f"❌ {name} failed with returncode {returncode}",
    #                 flush=True,
    #             )

    #     if failed:
    #         raise SystemExit(1)
    #     pass
    # finally:
    #     if tpu_monitor_started:
    #         stop_tpu_test_async()


if __name__ == "__main__":
    main()