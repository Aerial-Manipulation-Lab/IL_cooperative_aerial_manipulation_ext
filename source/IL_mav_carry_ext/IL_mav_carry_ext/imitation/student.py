# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student policy: raw policy obs -> action term's action."""

import torch

from .model import load_checkpoint


class StudentPolicy:
    """Flies a student network, trained or still training."""

    def __init__(self, model, spec):
        self.model, self.spec = model, spec

    @classmethod
    def load(cls, path, device):
        model, spec, _ = load_checkpoint(path, device=device)
        print(f"[INFO]: student from {path}")
        return cls(model, spec)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.spec.plan_to_action(self.plan(obs))

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """Each drone's whole plan, (N, num_drones, num_nodes, 12), env frame like the teacher's."""
        return self.spec.label_to_plan(self.model.predict(self.spec.build_inputs(obs)), obs)
