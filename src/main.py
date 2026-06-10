# src/main.py

import argparse
from pathlib import Path

from src.utils import load_config
from src.utils.device import get_device

from src.trainers import (
    run_train,
)


def build_config(command: str, overrides=None):

    config_dir = Path("configs")

    paths = [
        config_dir / "base.yaml",
    ]

    if command == "train":
        paths.append(
            config_dir / "train.yaml"
        )

    elif command == "eval":
        paths.append(
            config_dir / "eval.yaml"
        )

    return load_config(paths, overrides=overrides)


def main():

    parser = argparse.ArgumentParser(
        description="Adversarial MARL Cyber"
    )

    parser.add_argument(
        "command",
        choices=[
            "train",
            "eval",
        ],
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. --override train.device=cuda",
    )

    args = parser.parse_args()

    cfg = build_config(
        args.command,
        overrides=args.override,
    )

    requested_device = cfg.get("train", {}).get("device", "auto")

    device, device_status = get_device(requested_device)

    print(
        "\n=== CONFIGURATION LOADED ===\n"
    )

    print(cfg)

    print(
        f"\n=== TRAINING DEVICE: {device} ({device_status}) ===\n"
    )

    if args.command == "train":

        run_train(cfg, device=device)

    elif args.command == "eval":

        raise NotImplementedError(
            "Evaluation pipeline not implemented yet."
        )


if __name__ == "__main__":
    main()