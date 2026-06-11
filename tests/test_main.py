import os
import sys
import subprocess

subprocess.run(
    [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        "src.main",
        "train",

        "--override",
        "run.name=test_logs",

        "--override",
        "train.device=cpu",

        "--override",
        "train.resume=false",

        "--override",
        "train.episode_len=50",

        "--override",
        "train.episodes_per_update=5",

        "--override",
        "train.workers=5",

        "--override",
        "train.batch_size=25",

        "--override",
        "train.training_episodes=10",

        "--override",
        "train.epochs=4",
    ],
    check=True,
    env=os.environ.copy(),
)