# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The thesis student's inputs and labels (section 4.3; its `collab_drone_env._get_agent_obs` and `PINN_FE`).

Per drone, four streams, every position relative to the payload now (the load frame L: the payload's
position as origin, world axes), rotations as rot6d (the first two rows of the rotation matrix):

    load       2 s history: t, p_L, rot6d(R_L)                                   10 per sample
               with payload_velocity: t, p_L, v_L, w_L (body frame), rot6d(R_L)  16 per sample
    own drone  2 s history: t, p_i, v_i, w_i (body frame), rot6d(R_i)              16 per sample
    cables     2 s history: t, the cable's end on the load, its end on the drone   7 per sample
    reference  the MPC's nodes: t, p, v, a, w, alpha, rot6d(R)                    22 per node

t is the sample's time relative to now. The history is the thesis's 40 samples at 20 Hz.

Without history, the inputs are main's instead (`--model main`): the state now and the reference in main's
layout, so that main's student trains on the same flights. The payload's velocities only with payload_velocity,
which main always has.
"""

import torch

from ..plants import ROPE_END

PLAN_DIM = 12
"""Per horizon node: p(3), v(3), a(3), w(3), as `teacher_horizon` records it."""

HISTORY_STRIDE = 5
"""Every how many env steps of the recorded history the student sees a sample: 5 is the thesis's 20 Hz."""

HISTORY_TERMS = {
    "payload_position_hist": 3,
    "payload_orientation_hist": 9,
    "drone_positions_hist": 9,
    "drone_orientations_hist": 27,
    "drone_linear_velocities_hist": 9,
    "drone_angular_velocities_hist": 9,
}
"""The history obs terms and their size per env step."""

PAYLOAD_VELOCITY_TERMS = {"payload_linear_velocities_hist": 3, "payload_angular_velocities_hist": 3}
"""The payload's velocity history, read only with payload_velocity."""

STREAM_DIMS = {"load": 10, "drone": 16, "cables": 7, "reference": 22}
"""Values per sample of each input stream, in their order in the input vector."""

LOAD_DIM_WITH_VELOCITY = 16
"""The load stream's width with payload_velocity."""

NOW_SHARED_TERMS = ("payload_orientation",)
NOW_PAYLOAD_VELOCITY_TERMS = ("payload_linear_velocities", "payload_angular_velocities")
NOW_PER_DRONE_TERMS = (
    "drone_pos_rel_payload",
    "drone_orientations",
    "drone_linear_velocities",
    "drone_angular_velocities",
    "drone_cable_dir",
)
"""main's inputs besides the reference: shared, then drone-major (drone d gets block d), then the attach point."""

MAIN_REF_DIM = 18
"""main's reference per node: dp, v, a, rot6d (the first two columns of R_payload^T R_ref), w."""

LABEL_FRAMES = ("payload", "drone")
"""payload: the thesis's L, plan positions relative to the payload now, velocities as they are.
drone: plan positions and velocities relative to the drone's own now."""


def rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 9) row-major rotation matrices -> (..., 6): their first two rows, as pytorch3d's rot6d."""
    return matrix[..., :6]


class FeatureSpec:
    """Column layout of one dataset's obs, the plant behind its labels, and the label frame."""

    def __init__(self, meta: dict, label_frame: str = "payload", payload_velocity: bool = False, history: bool = True):
        if label_frame not in LABEL_FRAMES:
            raise ValueError(f"label_frame {label_frame!r} is not one of {LABEL_FRAMES}")
        self.meta, self.label_frame, self.payload_velocity = meta, label_frame, payload_velocity
        self.history_inputs = history
        self.now_shared = NOW_SHARED_TERMS + (NOW_PAYLOAD_VELOCITY_TERMS if payload_velocity else ())
        self.num_drones = meta["num_drones"]
        self.num_nodes = meta["num_nodes"]
        self.slices, self.dims, self.history = {}, {}, {}
        start = 0
        for term in meta["obs_terms"]:
            self.slices[term["name"]] = slice(start, start + term["dim"])
            self.dims[term["name"]] = term["dim"]
            self.history[term["name"]] = term.get("history_length", 0)
            start += term["dim"]
        self.obs_dim = start
        needed = [
            *HISTORY_TERMS,
            *(PAYLOAD_VELOCITY_TERMS if payload_velocity else ()),
            *(() if history else self.now_shared + NOW_PER_DRONE_TERMS),
            "payload_ref_horizon",
            "payload_position",
            "drone_pos_rel_payload",
            "drone_linear_velocities",
        ]
        missing = [t for t in needed if t not in self.slices]
        if missing:
            raise KeyError(f"obs terms missing from the dataset: {missing}")
        self.history_steps = self.history["payload_position_hist"]
        self.history_len = self.history_steps // HISTORY_STRIDE
        self.attach_points = torch.tensor([d["attach_point"] for d in meta["plant"]["drones"]], dtype=torch.float32)
        self.node_times = torch.tensor(meta["node_offsets_s"], dtype=torch.float32)
        # the sampled history's times relative to now, oldest first
        dt = HISTORY_STRIDE * meta["step_dt"]
        self.history_times = torch.arange(-(self.history_len - 1), 1, dtype=torch.float32) * dt

    @property
    def streams(self) -> list[tuple[str, int, int]]:
        """(name, samples, values per sample) of each input stream, in input order."""
        lengths = {"load": self.history_len, "drone": self.history_len, "cables": self.history_len}
        lengths["reference"] = self.num_nodes
        widths = {**STREAM_DIMS, "load": LOAD_DIM_WITH_VELOCITY if self.payload_velocity else STREAM_DIMS["load"]}
        return [(name, lengths[name], width) for name, width in widths.items()]

    @property
    def input_dim(self) -> int:
        if not self.history_inputs:
            shared = sum(self.dims[t] for t in self.now_shared) + self.num_nodes * MAIN_REF_DIM
            return shared + sum(self.dims[t] // self.num_drones for t in NOW_PER_DRONE_TERMS) + 3
        return sum(length * width for _, length, width in self.streams)

    @property
    def label_dim(self) -> int:
        return self.num_nodes * PLAN_DIM

    def _history(self, obs: torch.Tensor, term: str) -> torch.Tensor:
        """A history term's sampled values, (N, history_len, per-step size), oldest first."""
        size = {**HISTORY_TERMS, **PAYLOAD_VELOCITY_TERMS}[term]
        steps = obs[:, self.slices[term]].reshape(obs.shape[0], self.history_steps, size)
        return steps[:, HISTORY_STRIDE - 1 :: HISTORY_STRIDE]

    def build_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Flat policy obs (N, obs_dim) -> per-drone inputs (N, num_drones, input_dim)."""
        if not self.history_inputs:
            return self._inputs_now(obs)
        n, d, h, dev = obs.shape[0], self.num_drones, self.history_len, obs.device
        origin = obs[:, self.slices["payload_position"]][:, None, None]  # the payload now, (N, 1, 1, 3)

        load_p = self._history(obs, "payload_position_hist")[:, :, None]  # (N, H, 1, 3)
        load_r = self._history(obs, "payload_orientation_hist")[:, :, None]  # (N, H, 1, 9)
        drone_p = self._history(obs, "drone_positions_hist").reshape(n, h, d, 3)
        drone_r = self._history(obs, "drone_orientations_hist").reshape(n, h, d, 9)
        drone_v = self._history(obs, "drone_linear_velocities_hist").reshape(n, h, d, 3)
        drone_w = self._history(obs, "drone_angular_velocities_hist").reshape(n, h, d, 3)

        def rotate(matrix, vector):  # (.., 9) row-major R times (.., 3)
            return (matrix.unflatten(-1, (3, 3)) @ vector[..., None])[..., 0]

        def unrotate(matrix, vector):  # R^T times
            return (matrix.unflatten(-1, (3, 3)).transpose(-1, -2) @ vector[..., None])[..., 0]

        t = self.history_times.to(dev)[None, :, None, None].expand(n, h, d, 1)
        load = [load_p - origin]
        if self.payload_velocity:
            load_v = self._history(obs, "payload_linear_velocities_hist")[:, :, None]
            load_w = self._history(obs, "payload_angular_velocities_hist")[:, :, None]
            load += [load_v, unrotate(load_r, load_w)]
        load = torch.cat([*load, rot6d(load_r)], -1)
        load = load.expand(n, h, d, load.shape[-1])
        own = torch.cat([drone_p - origin, drone_v, unrotate(drone_r, drone_w), rot6d(drone_r)], -1)
        on_load = load_p + rotate(load_r.expand(n, h, d, 9), self.attach_points.to(dev).expand(n, h, d, 3))
        on_drone = drone_p + rotate(
            drone_r, torch.as_tensor(ROPE_END, dtype=torch.float32, device=dev).expand(n, h, d, 3)
        )
        cables = torch.cat([on_load - origin, on_drone - origin], -1)

        k = self.num_nodes
        ref = obs[:, self.slices["payload_ref_horizon"]].reshape(n, k, -1)
        ref = torch.cat([self.node_times.to(dev)[None, :, None].expand(n, k, 1), ref], -1)
        streams = [torch.cat([t, load], -1), torch.cat([t, own], -1), torch.cat([t, cables], -1)]
        flat = [s.transpose(1, 2).flatten(2) for s in streams]  # (N, d, H * width), sample-major
        return torch.cat([*flat, ref.flatten(1)[:, None].expand(n, d, -1)], -1)

    def _inputs_now(self, obs: torch.Tensor) -> torch.Tensor:
        """main's `build_inputs`, its reference block rebuilt from this branch's."""
        n, d, k = obs.shape[0], self.num_drones, self.num_nodes
        ref = obs[:, self.slices["payload_ref_horizon"]].reshape(n, k, -1)
        r0, r1 = ref[..., 15:18], ref[..., 18:21]  # rows of R_payload^T R_ref
        rel = torch.stack([r0, r1, torch.cross(r0, r1, dim=-1)], -2)
        columns = rel[..., :2].transpose(-1, -2).flatten(-2)
        ref = torch.cat([ref[..., 0:9], columns, ref[..., 9:12]], -1).flatten(1)
        shared = torch.cat([*(obs[:, self.slices[t]] for t in self.now_shared), ref], -1)
        own = torch.cat([obs[:, self.slices[t]].reshape(n, d, -1) for t in NOW_PER_DRONE_TERMS], -1)
        attach = self.attach_points.to(obs.device).expand(n, -1, -1)
        return torch.cat([shared[:, None].expand(-1, d, -1), own, attach], -1)

    def drone_state(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each drone's env-frame position and velocity now, (N, num_drones, 3) each."""
        n, d = obs.shape[0], self.num_drones
        payload = obs[:, self.slices["payload_position"]]
        position = payload[:, None] + obs[:, self.slices["drone_pos_rel_payload"]].reshape(n, d, 3)
        velocity = obs[:, self.slices["drone_linear_velocities"]].reshape(n, d, 3)
        return position, velocity

    def _origin(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What the labels' positions and velocities are relative to, (N, num_drones, 3) each."""
        if self.label_frame == "drone":
            return self.drone_state(obs)
        payload = obs[:, self.slices["payload_position"]][:, None].expand(-1, self.num_drones, -1)
        return payload, torch.zeros_like(payload)

    def plan_to_label(self, plan: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Teacher plan (N, num_drones, num_nodes, 12) -> labels (N, num_drones, label_dim), in `label_frame`."""
        position, velocity = self._origin(obs)
        label = plan.clone()
        label[..., 0:3] -= position[:, :, None]
        label[..., 3:6] -= velocity[:, :, None]
        return label.flatten(-2)

    def label_to_plan(self, label: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Inverse of `plan_to_label`: (N, num_drones, label_dim) -> (N, num_drones, num_nodes, 12)."""
        position, velocity = self._origin(obs)
        plan = label.reshape(*label.shape[:-1], self.num_nodes, PLAN_DIM).clone()
        plan[..., 0:3] += position[:, :, None]
        plan[..., 3:6] += velocity[:, :, None]
        return plan

    @staticmethod
    def plan_to_action(plan: torch.Tensor) -> torch.Tensor:
        """Plan (N, num_drones, num_nodes, 12) -> the action term's (N, num_drones * 12):
        node 1's p, v, a, w; w is the body-rate reference, world frame."""
        return plan[:, :, 1].flatten(1)
