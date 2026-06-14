# src/trainers/tpu.py
import os
import sys
import time
import subprocess
import threading


_TPU_TEST_PROCESS = None
_TPU_MONITOR_THREAD = None
_TPU_MONITOR_STOP = False


def _launch_tpu_test_process(
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
    global _TPU_TEST_PROCESS

    if _TPU_TEST_PROCESS is None:
        return None

    if _TPU_TEST_PROCESS.poll() is None:
        return None

    stdout, stderr = _TPU_TEST_PROCESS.communicate()
    returncode = _TPU_TEST_PROCESS.returncode

    _TPU_TEST_PROCESS = None

    if returncode == 0:
        # print("[TPU test] finished successfully", flush=True)
        return True

    # print("[TPU test] failed", flush=True)

    # if stderr:
        # print(stderr[-1000:], flush=True)

    return False


def _tpu_monitor_loop(
    interval_seconds=1800,
    batch=64,
    seq_len=512,
    hidden=1024,
    layers=12,
    heads=16,
    steps=100,
):
    global _TPU_MONITOR_STOP

    while not _TPU_MONITOR_STOP:
        check_tpu_test_async()

        started = _launch_tpu_test_process(
            batch=batch,
            seq_len=seq_len,
            hidden=hidden,
            layers=layers,
            heads=heads,
            steps=steps,
        )

        # if started:
        #     print("[TPU test] started", flush=True)

        time.sleep(interval_seconds)


def start_tpu_test_async(
    interval_seconds=1800,
    batch=64,
    seq_len=512,
    hidden=1024,
    layers=12,
    heads=16,
    steps=100,
):
    """
    Starts one background monitor thread.
    The thread launches a TPU test every interval_seconds.
    Non-blocking.
    """
    global _TPU_MONITOR_THREAD
    global _TPU_MONITOR_STOP

    if _TPU_MONITOR_THREAD is not None and _TPU_MONITOR_THREAD.is_alive():
        return False

    _TPU_MONITOR_STOP = False

    _TPU_MONITOR_THREAD = threading.Thread(
        target=_tpu_monitor_loop,
        kwargs=dict(
            interval_seconds=interval_seconds,
            batch=batch,
            seq_len=seq_len,
            hidden=hidden,
            layers=layers,
            heads=heads,
            steps=steps,
        ),
        daemon=True,
    )

    _TPU_MONITOR_THREAD.start()
    return True


def stop_tpu_test_async():
    global _TPU_MONITOR_STOP
    global _TPU_TEST_PROCESS

    _TPU_MONITOR_STOP = True

    if _TPU_TEST_PROCESS is not None and _TPU_TEST_PROCESS.poll() is None:
        _TPU_TEST_PROCESS.terminate()
        _TPU_TEST_PROCESS = None