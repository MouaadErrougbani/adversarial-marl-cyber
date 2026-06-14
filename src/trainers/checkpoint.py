# src/trainers/checkpoint.py

import os
import torch


def save_checkpoints(
    agents,
    checkpoint_dir,
    checkpoint_name,
    update_idx,
    episodes_per_update,
):
    """
    Save all PPO agents.

    Parameters
    ----------
    agents : list
        PPO agents.

    checkpoint_dir : str
        Checkpoint directory.

    checkpoint_name : str
        Run name.

    update_idx : int
        Current update index.

    episodes_per_update : int
        Episodes collected per PPO update.
    """

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    for agent_idx, agent in enumerate(
        agents
    ):

        latest_path = (
            f"{checkpoint_dir}/"
            f"{checkpoint_name}-"
            f"{agent_idx}_checkpoint.pt"
        )

        agent.save(
            latest_path
        )

        #
        # Long-term snapshots
        #
        if (
            update_idx % 10000
            < episodes_per_update
            and update_idx > episodes_per_update
        ):

            snapshot_path = (
                f"{checkpoint_dir}/"
                f"{checkpoint_name}-"
                f"{agent_idx}_"
                f"{update_idx // 1000}k.pt"
            )

            agent.save(
                snapshot_path
            )

def load_checkpoints(
    agents,
    checkpoint_dir,
    checkpoint_name,
):
    """
    Charge les checkpoints PPO pour tous les agents.

    Exemple:
        checkpoints/model-0_checkpoint.pt
        checkpoints/model-1_checkpoint.pt
        ...
    """

    for i, agent in enumerate(agents):

        ckpt_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_name}-{i}_checkpoint.pt",
        )

        if os.path.exists(ckpt_path):

            agent.load_weights(
                ckpt_path
            )


        else:

            print(
                f"[Checkpoint] "
                f"Agent {i}: "
                f"file not found "
                f"({ckpt_path})", flush=True
            )

def save_logs(
    log,
    log_dir,
    run_name,
):
    """
    Save training logs.

    Parameters
    ----------
    log : list
        Training history.

    log_dir : str
        Logs directory.

    run_name : str
        Experiment name.
    """

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    log_path = os.path.join(
        log_dir,
        f"{run_name}.pt",
    )

    torch.save(
        log,
        log_path,
    )

    return log_path

def load_logs(
    log_dir,
    resume=False,
    resume_name=None,
    override_start_iter=None,
):
    """
    Charge les logs d'entraînement
    et calcule start_iter.

    New log format:
        list[dict]

    Old log format:
        list[tuple] = [(avg_reward, update, avg_loss), ...]
    """

    log = []
    start_iter = 0

    if resume and resume_name:

        log_path = os.path.join(
            log_dir,
            f"{resume_name}.pt",
        )

        if os.path.exists(log_path):

            loaded_log = torch.load(
                log_path,
                map_location="cpu",
            )

            if not isinstance(loaded_log, list):
                print(
                    f"Warning: invalid log format in {log_path}. "
                    f"Expected list, got {type(loaded_log)}. "
                    f"Starting with empty log.", flush=True
                )

                log = []
                start_iter = 0

            else:
                log = loaded_log
                start_iter = len(log)

                if len(log) > 0:
                    first_entry = log[0]

                    if isinstance(first_entry, tuple):
                        print(
                            "Warning: old tuple log format detected. "
                            "It is recommended to start a new run "
                            "or convert old logs before resuming.",
                            flush=True
                        )

                    elif isinstance(first_entry, dict):
                        print(
                            "New dict log format detected.",
                            flush=True
                        )

                    else:
                        print(
                            f"Warning: unknown log entry type: "
                            f"{type(first_entry)}",
                            flush=True
                        )

                print(
                    f"Resume enabled: "
                    f"loaded log {log_path} "
                    f"(start_iter={start_iter})"
                    , flush=True
                )

        else:

            print(
                f"Warning: log file not found: "
                f"{log_path}", flush=True
            )

    if override_start_iter is not None:

        start_iter = int(
            override_start_iter
        )

        print(
            f"Resume override "
            f"start_iter={start_iter}", flush=True
        )

    return log, start_iter

