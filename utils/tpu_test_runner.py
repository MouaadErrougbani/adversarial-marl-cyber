# utils/tpu_test_runner.py

import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    # مهم: قبل import torch_xla
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    import torch
    import torch_xla.core.xla_model as xm

    print("🔍 Checking TPU/XLA...", flush=True)

    device = xm.xla_device()
    device_str = str(device)

    if not device_str.startswith("xla"):
        print(f"⚠️ XLA device non détecté: {device_str}", flush=True)
        return 1

    print("🔍 Testing TPU/XLA...", flush=True)
    print("Device:", device, flush=True)

    x = torch.ones((4, 4), device=device)
    y = x @ x
    xm.mark_step()

    print("Small tensor device:", y.device, flush=True)
    print(y.cpu(), flush=True)

    size = args.size
    steps = args.steps

    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    t0 = time.time()
    c = a @ b
    xm.mark_step()

    # مهم: نجبر الحساب يكمل فعلاً
    _ = c[0, 0].detach().cpu().item()

    warmup_time = time.time() - t0

    t0 = time.time()

    for _ in range(steps):
        c = a @ b
        xm.mark_step()

    result = c[0, 0].detach().cpu().item()

    total_time = time.time() - t0
    avg_time = total_time / steps

    print(f"Warmup time: {warmup_time:.2f}s", flush=True)
    print(f"Benchmark steps: {steps}", flush=True)
    print(f"Matrix size: {size}x{size}", flush=True)
    print(f"Total time: {total_time:.2f}s", flush=True)
    print(f"Average time per step: {avg_time:.4f}s", flush=True)
    print("Result sample:", result, flush=True)

    print("✅ TPU test finished successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())