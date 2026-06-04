import argparse

from utils.config import load_config, merge_dicts
from pipelines.train import run_train, default_config
from pipelines.eval import run_eval


def build_parser():
    parser = argparse.ArgumentParser("Project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--config",
        action="append",
        default=["config/base.yaml", "config/train.yaml"],
        help="Path to YAML config (repeatable)",
    )
    train_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (key=value)",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--config",
        action="append",
        default=["config/base.yaml", "config/eval.yaml"],
        help="Path to YAML config (repeatable)",
    )
    eval_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (key=value)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    if args.command == "train":
        base_cfg = default_config()
        merge_dicts(base_cfg, cfg)
        run_train(base_cfg)
    else:
        run_eval(cfg)


if __name__ == "__main__":
    main()
