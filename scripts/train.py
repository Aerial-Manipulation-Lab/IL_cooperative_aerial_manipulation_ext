# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the uniform per-drone student on NMPC demonstrations (behaviour cloning).

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/train.py logs/record/mpc_demos'
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from IL_mav_carry_ext.imitation import (
    PLAN_DIM,
    BehaviourCloning,
    build_model,
    load_episodes,
    save_checkpoint,
)
from IL_mav_carry_ext.reporting import TensorBoardReporter, TextReporter

parser = argparse.ArgumentParser(description="Behaviour cloning on NMPC demonstrations.")
parser.add_argument("datasets", nargs="+", help="Dataset paths, with or without .hdf5.")
parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of episodes held out.")
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for the split and the initialisation."
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--model", choices=["mlp"], default="mlp", help="Student network, see build_model."
)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--log_dir", type=str, default="logs/bc")
parser.add_argument("--run_name", type=str, default=None, help="Defaults to a timestamp.")
args = parser.parse_args()


def split(episodes, val_frac, seed):
    """Stack into per-drone samples, split by episode: (x_train, y_train, x_val, y_val)."""
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, round(val_frac * len(episodes)))
    if n_val >= len(episodes):
        raise ValueError(f"need more than {n_val} episodes to hold {n_val} out for validation")

    def stack(ids):
        x = torch.cat([episodes[i][0] for i in ids]).flatten(0, 1)
        y = torch.cat([episodes[i][1] for i in ids]).flatten(0, 1)
        return x, y

    return (*stack(order[n_val:]), *stack(order[:n_val]))


def main():
    report = TextReporter("train")
    torch.manual_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.log_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    episodes, spec = load_episodes(args.datasets)
    report.info(f"loaded {len(episodes)} episodes from {len(args.datasets)} datasets")
    x_train, y_train, x_val, y_val = (
        t.to(args.device) for t in split(episodes, args.val_frac, args.seed)
    )
    del episodes
    report.info(
        f"{len(x_train)} train / {len(x_val)} val samples, "
        f"input {x_train.shape[-1]}, label {y_train.shape[-1]} ({spec.num_nodes} nodes x {PLAN_DIM})"
    )

    model = build_model(spec, args.model).to(args.device)
    bc = BehaviourCloning(model, (x_train, y_train), (x_val, y_val), args.lr, args.batch_size)

    config = {
        **vars(args),
        "run_name": run_name,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tensorboard = TensorBoardReporter(run_dir)
    report.info(f"logging to {run_dir}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        metrics = {**bc.update(), "epoch_s": time.time() - start}
        report.metrics(f"epoch {epoch}", metrics)
        tensorboard.metrics(metrics, epoch)

        extra = {"epoch": epoch, "metrics": metrics, "train_args": vars(args)}
        save_checkpoint(os.path.join(run_dir, "last.pt"), model, spec, **extra)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, spec, **extra)

    tensorboard.close()
    report.metrics("done", {"best_val_loss": best})
    report.info(f"checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
