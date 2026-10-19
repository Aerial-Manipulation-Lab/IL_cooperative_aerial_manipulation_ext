# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Everything around the student, shared by the scripts: building, saving and loading the student, its
loss, and the recorded datasets."""

import json
import os

import h5py
import torch

from .features import PLAN_DIM, PLAN_PARTS, FeatureSpec
from .models import BCPolicy


def build_model(spec: FeatureSpec, name: str) -> BCPolicy:
    """An untrained student for the rows `spec` describes."""
    if name == "mlp":
        return BCPolicy(spec.input_dim, spec.label_dim, hidden=(256, 256, 256))
    raise ValueError(f"unknown student {name!r}")


def save_checkpoint(path, model: BCPolicy, spec: FeatureSpec, **extra):
    """Everything needed to fly the student without the training script."""
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
    """(model, the rows it was trained on, the whole checkpoint dict) from `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    model = BCPolicy(cfg["input_dim"], cfg["label_dim"], tuple(cfg["hidden"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, FeatureSpec.from_dict(ckpt["meta"]), ckpt


def bc_loss(model, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE between the student's prediction and the teacher's label, in physical units as in the thesis."""
    return torch.nn.functional.mse_loss(model(inputs), labels)


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
    """The student on examples it hasn't trained on: the loss, and the plan's errors in physical units."""
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
