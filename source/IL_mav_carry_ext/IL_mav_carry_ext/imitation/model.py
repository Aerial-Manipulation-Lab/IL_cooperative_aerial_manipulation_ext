# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The behaviour-cloning students: an MLP and a physics-informed TCN + FCN.

Both map a drone's policy row to its label row, its plan in its own frame, which is also its
action. Physical units, unnormalised, as the thesis trains them."""

import torch
from torch import nn

from .features import input_dim, label_dim


class BCPolicy(nn.Module):
    """A drone's policy row -> its label row."""

    def __init__(self, input_dim: int, label_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.cfg = {"input_dim": input_dim, "label_dim": label_dim, "hidden": list(hidden)}
        layers, width = [], input_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ELU()]
            width = h
        layers.append(nn.Linear(width, label_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inputs (..., input_dim) -> labels (..., label_dim)."""
        return self.net(inputs)


def build_model(meta: dict, name: str) -> BCPolicy:
    """An untrained student for the policy rows `meta` describes."""
    if name == "mlp":
        return BCPolicy(input_dim(meta), label_dim(meta), hidden=(256, 256, 256))
    raise ValueError(f"unknown student {name!r}")


def save_checkpoint(path, model: BCPolicy, meta: dict, **extra):
    """Everything needed to fly the student without the training script."""
    torch.save(
        {
            "model": type(model).__name__,
            "model_cfg": model.cfg,
            "state_dict": model.state_dict(),
            "meta": meta,
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[BCPolicy, dict, dict]:
    """(model, the policy rows' metadata, the whole checkpoint dict) from `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    model = BCPolicy(cfg["input_dim"], cfg["label_dim"], tuple(cfg["hidden"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["meta"], ckpt
