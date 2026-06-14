# utils/tpu_test_runner.py

import os
import time
import argparse

def run(args, xm, optimizer, head, model, loss_fn, x, target):

    t0 = time.time()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        y = head(model(x))
        loss = loss_fn(y, target)

        loss.backward()
        optimizer.step()

        xm.mark_step()

        if step % 10 == 0:
            print(f"[TPU test] Step {step}, Loss: {loss.detach().cpu().item()}", flush=True)

    final_loss = loss.detach().cpu().item()
    total_time = time.time() - t0

    print(f"[TPU test] Final loss: {final_loss}, Total time: {total_time}", flush=True)
    print("=="*40, flush=True)
    return 0



def main():
   
    print("[TPU test] tpu_test_runner starting", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print("=="*40, flush=True) 
     # Important avant import torch_xla
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    print("[TPU test] Importing torch and torch_xla", flush=True)
    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm
    print("[TPU test] torch and torch_xla imported", flush=True)

    device = xm.xla_device()
    device_str = str(device)
    print("[TPU test] Device tpu_test_runner : ", device_str, flush=True)
    if not device_str.startswith("xla"):
        return 1

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
            ok = run(args, xm, optimizer, head, model, loss_fn, x, target)

            print(
                f"TPU test finished with code {ok}",
                flush=True
            )

        except Exception as e:
            print(
                f"TPU test crashed: {e}",
                flush=True
            )

        time.sleep(5 * 60)

      


if __name__ == "__main__":
    raise SystemExit(main())