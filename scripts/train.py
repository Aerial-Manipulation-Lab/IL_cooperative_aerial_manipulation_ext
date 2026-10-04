# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the uniform per-drone student on NMPC demonstrations (behaviour cloning).

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/train.py datasets/mpc_demos'
"""

import argparse
import json
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch
from IL_mav_carry_ext.imitation import PLAN_DIM, BCPolicy, FeatureSpec, PINPolicy, save_checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Behaviour cloning on NMPC demonstrations.")
parser.add_argument("datasets", nargs="+", help="Dataset paths, with or without .hdf5.")
parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of episodes held out.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the split and the initialisation.")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--model", choices=["mlp", "pin"], default="mlp", help="BCPolicy (mlp) or the physics-informed PINPolicy (pin)."
)
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256], help="Hidden layer widths.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--log_dir", type=str, default="logs/bc")
parser.add_argument("--run_name", type=str, default=None, help="Defaults to a timestamp.")
args = parser.parse_args()


def load_file(stem, spec):
    """Usable (inputs, labels) episodes from one dataset file, as (steps, num_drones, dim) CPU tensors."""
    episodes, skipped = [], 0
    with h5py.File(stem + ".hdf5", "r") as f:
        print(f"[INFO]: loading {len(f['data'])} episodes from {stem}.hdf5", flush=True)
        for n, name in enumerate(f["data"], start=1):
            if n % 50 == 0:
                print(f"[INFO]:   {n}/{len(f['data'])} episodes", flush=True)
            ep = f["data"][name]
            if not bool(ep.attrs.get("success", False)):
                skipped += 1
                continue
            ok = ep["mpc_ok"][()].astype(bool)
            obs = torch.as_tensor(ep["obs"][()][ok], dtype=torch.float32)
            plan = torch.as_tensor(ep["teacher_horizon"][()][ok], dtype=torch.float32)
            episodes.append((spec.build_inputs(obs), spec.plan_to_label(plan, obs)))
    return episodes, skipped


def load_episodes(paths):
    """[(inputs, labels)] per usable episode over all files, and the shared spec."""
    spec, episodes, skipped = None, [], 0
    for path in paths:
        stem = os.path.splitext(path)[0]
        with open(stem + ".meta.json") as f:
            meta = json.load(f)
        if spec is None:
            spec = FeatureSpec(meta)
        elif meta["obs_terms"] != spec.meta["obs_terms"]:
            raise ValueError(f"{stem}: obs layout differs from {paths[0]}")
        file_episodes, file_skipped = load_file(stem, spec)
        episodes.extend(file_episodes)
        skipped += file_skipped
    print(f"[INFO]: {len(episodes)} episodes used, {skipped} unsuccessful skipped")
    return episodes, spec


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


@torch.no_grad()
def evaluate(model, x, y, spec, batch_size):
    """Validation loss (normalised MSE) and errors in physical units."""
    loss, sq_err = 0.0, torch.zeros(spec.num_nodes, PLAN_DIM, device=x.device)
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        pred = model(model.normalise_inputs(xb))
        loss += torch.nn.functional.mse_loss(pred, model.normalise_labels(yb), reduction="sum").item()
        pred_phys = pred * model.out_std + model.out_mean
        sq_err += ((pred_phys - yb) ** 2).reshape(-1, spec.num_nodes, PLAN_DIM).sum(0)
    rmse = (sq_err / len(x)).sqrt()

    def vec(node, part):
        return float(rmse[node, part].norm())

    p, v, a, w = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
    return {
        "val/loss": loss / y.numel(),
        "val/node1_pos_cm": 100 * vec(1, p),
        "val/node1_vel_mps": vec(1, v),
        "val/node1_acc_mps2": vec(1, a),
        "val/node1_rate_radps": vec(1, w),
        "val/last_node_pos_cm": 100 * vec(-1, p),
        "val/horizon_pos_cm": 100 * float(rmse[:, p].norm(dim=-1).mean()),
    }


def main():
    torch.manual_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.log_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    episodes, spec = load_episodes(args.datasets)
    x_train, y_train, x_val, y_val = (t.to(args.device) for t in split(episodes, args.val_frac, args.seed))
    del episodes
    print(
        f"[INFO]: {len(x_train)} train / {len(x_val)} val samples, "
        f"input {spec.input_dim}, label {spec.label_dim} ({spec.num_nodes} nodes x {PLAN_DIM})"
    )

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    if args.model == "pin":
        model = PINPolicy.from_spec(spec, tuple(args.hidden)).to(args.device)
    else:
        model = BCPolicy(spec.input_dim, spec.label_dim, tuple(args.hidden)).to(args.device)
    model.set_normalisation(x_train, y_train)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5)

    config = {**vars(args), "run_name": run_name, "train_samples": len(x_train), "val_samples": len(x_val)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    writer = SummaryWriter(run_dir)
    print(f"[INFO]: logging to {run_dir} (tensorboard)")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            pred = model(model.normalise_inputs(xb))
            loss = torch.nn.functional.mse_loss(pred, model.normalise_labels(yb))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        metrics = evaluate(model, x_val, y_val, spec, args.batch_size)
        metrics["train/loss"] = train_loss / len(x_train)
        metrics["lr"] = optimiser.param_groups[0]["lr"]
        scheduler.step(metrics["val/loss"])
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)

        extra = {"epoch": epoch, "metrics": metrics, "train_args": vars(args)}
        save_checkpoint(os.path.join(run_dir, "last.pt"), model, spec, **extra)
        if metrics["val/loss"] < best:
            best = metrics["val/loss"]
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, spec, **extra)
        print(
            f"[{epoch:3d}] train {metrics['train/loss']:.4f}  val {metrics['val/loss']:.4f}  "
            f"node1 pos {metrics['val/node1_pos_cm']:.2f} cm  vel {metrics['val/node1_vel_mps']:.3f} m/s  "
            f"acc {metrics['val/node1_acc_mps2']:.3f} m/s2  | horizon pos {metrics['val/horizon_pos_cm']:.2f} cm  "
            f"({time.time() - start:.1f}s)",
            flush=True,
        )

    writer.close()
    print(f"[INFO]: best val loss {best:.4f}; checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
