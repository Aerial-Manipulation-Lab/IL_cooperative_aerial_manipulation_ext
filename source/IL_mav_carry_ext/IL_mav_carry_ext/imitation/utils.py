# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import json
import os

import h5py
import torch

from .features import PLAN_DIM, PLAN_PARTS, FeatureSpec

# TODO: Rename BCPolicy to more fitting name. Suggest names first
from .models import BCPolicy


def build_model(spec: FeatureSpec, name: str) -> BCPolicy:
    """Builds a untrained student model based on the input/label spec given."""
    # when more models become available we add them here
    if name == "mlp":
        return BCPolicy(spec.input_dim, spec.label_dim, hidden=(256, 256, 256))
    raise ValueError(f"unknown student {name!r}")


# TODO: rename save_checkpoint to save_model_checkpoint and load_checkpoint tp load_model_checkpoint
def save_checkpoint(path, model: BCPolicy, spec: FeatureSpec, **extra):
    """Save everything needed to fly the student without the training script."""
    torch.save(
        {
            "model": type(model).__name__,
            "model_cfg": model.cfg,
            "state_dict": model.state_dict(),
            "meta": spec.to_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[BCPolicy, FeatureSpec, dict]:
    """Loads the checkpoint of a model. As saved by save_model_checkpoint"""
    model_checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = model_checkpoint["model_cfg"]
    model = BCPolicy(cfg["input_dim"], cfg["label_dim"], tuple(cfg["hidden"])).to(device)
    model.load_state_dict(model_checkpoint["state_dict"])
    model.eval()
    return model, FeatureSpec.from_dict(model_checkpoint["meta"]), model_checkpoint


# TODO: rename this to something more descriptivel. This is basically our place to change the loss function
def bc_loss(model, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Returns the loss we use in the"""
    return torch.nn.functional.mse_loss(model(inputs), labels)


# TODO: buffer is also a vague class name
class Buffer:
    """Every (policy row, label row) example so far, one row per drone and step."""

    def __init__(self):
        self.xs, self.ys = [], []

    def add(self, inputs, labels):
        self.xs.append(inputs.flatten(0, -2))
        self.ys.append(labels.flatten(0, -2))

    def __len__(self):
        return sum(len(x) for x in self.xs)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All examples so far, oldest first: (inputs, labels)."""
        return torch.cat(self.xs), torch.cat(self.ys)


def fit(model, optimiser, x: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int) -> float:
    """`epochs` passes over every example, in random order, one optimiser step per batch; the mean `bc_loss`."""
    model.train()
    loss_sum = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), batch_size):
            idx = order[i : i + batch_size]
            loss = bc_loss(model, x[idx], y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(idx)
    return loss_sum / (epochs * len(x))


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, batch_size: int = 4096) -> dict[str, float]:
    """Evaluate the student model on unseen examples."""
    model.eval()
    num_nodes = y.shape[-1] // PLAN_DIM
    loss, sq_err = 0.0, torch.zeros(num_nodes, PLAN_DIM, device=x.device)
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        pred = model(xb)
        loss += torch.nn.functional.mse_loss(pred, yb, reduction="sum").item()
        sq_err += ((pred - yb) ** 2).reshape(-1, num_nodes, PLAN_DIM).sum(0)
    rmse = (sq_err / len(x)).sqrt()

    def vec(node, part):
        return float(rmse[node, part].norm())

    p, v, a, w = PLAN_PARTS.values()
    return {
        "val_loss": loss / y.numel(),
        "node1_pos_err_cm": 100 * vec(1, p),
        "node1_vel_err_mps": vec(1, v),
        "node1_acc_err_mps2": vec(1, a),
        "node1_rate_err_radps": vec(1, w),
        "last_node_pos_err_cm": 100 * vec(-1, p),
        "horizon_pos_err_cm": 100 * float(rmse[:, p].norm(dim=-1).mean()),
    }


def load_episodes(paths: list[str]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], FeatureSpec]:
    """(policy rows, label rows) per episode over all files, as (steps, num_drones, dim) CPU tensors, and their spec.

    Every step whose solve succeeded is kept, also from episodes that ended early: under DAgger those
    are where the student crashed. Episodes without steps (reset again right after a reset) are skipped.
    """
    spec, episodes = None, []
    for path in paths:
        stem = os.path.splitext(path)[0]
        with open(stem + ".meta.json") as f:
            file_spec = FeatureSpec.from_dict(json.load(f))
        if spec is None:
            spec = file_spec
        elif file_spec != spec:
            raise ValueError(f"{stem}: its rows differ from {paths[0]}'s")
        with h5py.File(stem + ".hdf5", "r") as f:
            for ep in f["data"].values():
                if ep.attrs["num_samples"] == 0:
                    continue
                ok = ep["mpc_ok"][()].astype(bool)
                obs = torch.as_tensor(ep["obs"][()][ok], dtype=torch.float32)
                label = torch.as_tensor(ep["label"][()][ok], dtype=torch.float32)
                episodes.append((obs, label))
    return episodes, spec
