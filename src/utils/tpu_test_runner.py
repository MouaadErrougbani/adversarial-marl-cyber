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

    final_loss = loss.detach().cpu().item()
    total_time = time.time() - t0

    return 0

def main(args):
   
    

    # Important avant import torch_xla
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    device_str = str(device)
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
            run(args, xm, optimizer, head, model, loss_fn, x, target)


        except Exception as e:
            print(
                f"TPU test crashed: {e}",
                flush=True
            )

        

      
