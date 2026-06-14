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


        time.sleep(interval_seconds)



def start_tpu_test_async(
    batch=64,
    seq_len=512,
    hidden=1024,
    layers=12,
    heads=16,
    steps=1,
):
    from src.utils.tpu_test_runner import run
    import threading
    from types import SimpleNamespace
    import os
    import time

    args = SimpleNamespace(
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
        layers=layers,
        heads=heads,
        steps=steps,
    )

    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    device_str = str(device)


    encoder_layer = nn.TransformerEncoderLayer(
        d_model=args.hidden,
        nhead=args.heads,
        dim_feedforward=args.hidden * 4,
        dropout=0.0,
        batch_first=True,
        activation="gelu",
    )

    model = nn.TransformerEncoder(
        encoder_layer,
        num_layers=args.layers,
    ).to(device)

    head = nn.Linear(args.hidden, args.hidden).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=1e-4,
    )

    model.train()
    head.train()

    x = torch.randn(
            args.batch,
            args.seq_len,
            args.hidden,
            device=device,
        )

    target = torch.randn(
            args.batch,
            args.seq_len,
            args.hidden,
            device=device,
        )

    loss_fn = nn.MSELoss()

        # Warmup
    optimizer.zero_grad(set_to_none=True)
    y = head(model(x))
    loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()
    xm.mark_step()

    _ = loss.detach().cpu().item()


   
    
    while True:
        try:
            threading.Thread(
                target=run,
                args=(args, xm, optimizer, head, model, loss_fn, x, target),
                daemon=True,
            ).start()

        except Exception as e:
            print(
                f"TPU test crashed: {e}",
                flush=True
            )

        time.sleep(5 * 60)







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







def stop_tpu_test_async():
    global _TPU_MONITOR_STOP
    global _TPU_TEST_PROCESS

    _TPU_MONITOR_STOP = True

    if _TPU_TEST_PROCESS is not None and _TPU_TEST_PROCESS.poll() is None:
        _TPU_TEST_PROCESS.terminate()
        _TPU_TEST_PROCESS = None