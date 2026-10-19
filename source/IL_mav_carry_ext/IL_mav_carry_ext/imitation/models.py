# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The behaviour-cloning student: an MLP from a drone's policy row to its label row, its plan in its
own frame, which is also its action. Physical units, unnormalised, as the thesis trains it."""

import torch
from torch import nn


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
