# utils/tpu_test_runner.py

import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    # Important avant import torch_xla
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm

    print("🔥 TPU Transformer load starting...", flush=True)

    device = xm.xla_device()
    device_str = str(device)

    if not device_str.startswith("xla"):
        print(f"⚠️ XLA device non détecté: {device_str}", flush=True)
        return 1

    print("Device:", device, flush=True)
    print(
        f"batch={args.batch}, seq_len={args.seq_len}, hidden={args.hidden}, "
        f"layers={args.layers}, heads={args.heads}, steps={args.steps}",
        flush=True,
    )

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

    t0 = time.time()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        y = head(model(x))
        loss = loss_fn(y, target)

        loss.backward()
        optimizer.step()

        xm.mark_step()

        if step % 10 == 0:
            loss_value = loss.detach().cpu().item()
            print(f"step={step}/{args.steps}, loss={loss_value:.6f}", flush=True)

    final_loss = loss.detach().cpu().item()
    total_time = time.time() - t0

    print(f"Total TPU model load time: {total_time:.2f}s", flush=True)
    print(f"Final loss: {final_loss:.6f}", flush=True)
    print("✅ TPU Transformer load finished", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())