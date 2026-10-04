# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-loaded student policy: raw policy obs -> action term's action."""

import torch

from .model import load_checkpoint


class StudentPolicy:
    """Flies the checkpoint: maps the policy obs to the action term's action."""

    def __init__(self, path, device):
        self.model, self.spec, checkpoint = load_checkpoint(path, device=device)
        print(
            f"[INFO]: student from {path} (epoch {checkpoint['epoch']}, "
            f"val node-1 pos {checkpoint['metrics']['val/node1_pos_cm']:.2f} cm)"
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        label = self.model.predict(self.spec.build_inputs(obs))
        return self.spec.plan_to_action(self.spec.label_to_plan(label, obs))
