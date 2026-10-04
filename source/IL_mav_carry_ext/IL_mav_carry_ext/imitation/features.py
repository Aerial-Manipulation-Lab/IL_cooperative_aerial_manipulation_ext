# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

SHARED_TERMS = (
    "payload_orientation",
    "payload_linear_velocities",
    "payload_angular_velocities",
    "payload_ref_horizon",
)
"""Obs terms every drone gets a copy of."""

PER_DRONE_TERMS = (
    "drone_pos_rel_payload",
    "drone_orientations",
    "drone_linear_velocities",
    "drone_angular_velocities",
    "drone_cable_dir",
)
"""Drone-major obs terms; drone d gets only block d of each."""

PLAN_DIM = 12
"""Per horizon node: p(3), v(3), a(3), w(3), as `teacher_horizon` records it."""

LABEL_FRAME = "drone"
"""What the label is relative to; stored in every checkpoint and checked on load."""


class FeatureSpec:
    """Column layout of one dataset's obs, and the plant behind its labels."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.num_drones = meta["num_drones"]
        self.num_nodes = meta["num_nodes"]
        self.slices, self.dims = {}, {}
        start = 0

        for term in meta["obs_terms"]:
            self.slices[term["name"]] = slice(start, start + term["dim"])
            self.dims[term["name"]] = term["dim"]
            start += term["dim"]
        self.obs_dim = start
        missing = [t for t in SHARED_TERMS + PER_DRONE_TERMS + ("payload_position",) if t not in self.slices]
        if missing:
            raise KeyError(f"obs terms missing from the dataset: {missing}")
        self.attach_points = torch.tensor([d["attach_point"] for d in meta["plant"]["drones"]], dtype=torch.float32)

    @property
    def input_dim(self) -> int:
        shared = sum(self.dims[t] for t in SHARED_TERMS)
        own = sum(self.dims[t] // self.num_drones for t in PER_DRONE_TERMS)
        return shared + own + 3  # attach point appended last

    @property
    def label_dim(self) -> int:
        return self.num_nodes * PLAN_DIM

    def input_slice(self, term: str) -> slice:
        """Where `term` sits in one drone's `build_inputs` vector."""
        widths = [(t, self.dims[t]) for t in SHARED_TERMS]
        widths += [(t, self.dims[t] // self.num_drones) for t in PER_DRONE_TERMS]
        start = 0
        for name, width in widths:
            if name == term:
                return slice(start, start + width)
            start += width
        raise KeyError(f"{term} is not a student input term")

    def build_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Flat policy obs (N, obs_dim) -> per-drone inputs (N, num_drones, input_dim)."""
        n, d = obs.shape[0], self.num_drones
        shared = torch.cat([obs[:, self.slices[t]] for t in SHARED_TERMS], dim=-1)
        own = torch.cat([obs[:, self.slices[t]].reshape(n, d, -1) for t in PER_DRONE_TERMS], dim=-1)
        attach = self.attach_points.to(obs.device).expand(n, -1, -1)
        return torch.cat([shared[:, None].expand(-1, d, -1), own, attach], dim=-1)

    def drone_state(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each drone's env-frame position and velocity now, (N, num_drones, 3) each."""
        n, d = obs.shape[0], self.num_drones
        payload = obs[:, self.slices["payload_position"]]
        position = payload[:, None] + obs[:, self.slices["drone_pos_rel_payload"]].reshape(n, d, 3)
        velocity = obs[:, self.slices["drone_linear_velocities"]].reshape(n, d, 3)
        return position, velocity

    def plan_to_label(self, plan: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Teacher plan (N, num_drones, num_nodes, 12) -> labels (N, num_drones, label_dim),
        relative to each drone's own state now (`LABEL_FRAME`)."""
        position, velocity = self.drone_state(obs)
        label = plan.clone()
        label[..., 0:3] -= position[:, :, None]
        label[..., 3:6] -= velocity[:, :, None]
        return label.flatten(-2)

    def label_to_plan(self, label: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Inverse of `plan_to_label`: (N, num_drones, label_dim) -> (N, num_drones, num_nodes, 12)."""
        position, velocity = self.drone_state(obs)
        plan = label.reshape(*label.shape[:-1], self.num_nodes, PLAN_DIM).clone()
        plan[..., 0:3] += position[:, :, None]
        plan[..., 3:6] += velocity[:, :, None]
        return plan

    @staticmethod
    def plan_to_action(plan: torch.Tensor) -> torch.Tensor:
        """Plan (N, num_drones, num_nodes, 12) -> the action term's (N, num_drones * 12):
        node 1's p, v, a with zeros after."""
        node1 = plan[:, :, 1]
        action = torch.cat([node1[..., :9], torch.zeros_like(node1[..., 9:])], dim=-1)
        return action.flatten(1)
