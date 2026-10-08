# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms: a payload reference per env, and the NMPC teacher tracking it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import python_mpc_cusadi
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import (
    configclass,  # explicit: isaaclab.utils lazy-exports this name and it can be shadowed by the submodule
)
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, sample_uniform
from python_mpc_cusadi import Approach, FigureEight, LoadState, LoadTrajectory
from scipy.spatial.transform import Rotation

from ....mpc.teacher import MpcTeacher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REFERENCE_DATA = Path(python_mpc_cusadi.__file__).parents[2] / "data"
"""Where the reference trajectories live, in python_mpc_cusadi."""


class MpcCommand(CommandTerm):
    """A payload reference per env, built at reset from the payload's pose, and the NMPC teacher tracking it.

    Subclasses say which reference (`_build_reference`). On every reset it is
    rebuilt and handed to that env's MPC, which solves once for the fresh
    state; `_update_command` solves the rest. So every env solves once per
    step, at `common_step_counter * step_dt`, the clock the references are
    stamped with. The command is the reference pose now: (x, y, z, qx, qy, qz,
    qw), env frame.
    """

    cfg: MpcCommandCfg

    def __init__(self, cfg: MpcCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 6] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.references: list = [None] * self.num_envs
        self.teacher = MpcTeacher(env, rebuild=cfg.rebuild)
        self.solved_at = np.full(self.num_envs, -1)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_w

    def get_reference(self, env_id: int) -> LoadTrajectory:
        """The reference for one env, as the MPC wants it."""
        return self.references[int(env_id)]

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        """The reference for one env, leaving `start` (the payload at rest now) at `start.time`."""
        raise NotImplementedError

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids].tolist()
        measured = self.robot.data.body_com_state_w.torch[ids, self.body_idx].cpu().numpy()
        origins = self._env.scene.env_origins[ids].cpu().numpy()
        for j, i in enumerate(ids):
            start = LoadState(time=self._now(), p=measured[j, :3] - origins[j], q=measured[j, 3:7])
            self.references[i] = self._build_reference(i, start)
            self.teacher.reset(i, self.references[i])
        for _ in range(self.cfg.warmup_solves if self.cfg.solve_teacher else 0):
            self._solve(ids)
        self._track(ids)

    def _update_command(self):
        if self.cfg.solve_teacher:
            self._solve(np.flatnonzero(self.solved_at != self._env.common_step_counter).tolist())
        self._track(range(self.num_envs))

    def _update_metrics(self):
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_com_state_w.torch[:, self.body_idx, :3] - self._env.scene.env_origins,
            self.robot.data.body_com_state_w.torch[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _solve(self, env_ids: list[int]):
        self.teacher.solve(env_ids, self._now())
        self.solved_at[env_ids] = self._env.common_step_counter

    def _track(self, env_ids):
        """The command: each reference's pose now."""
        ids = list(env_ids)
        now = [self.references[i].window(self._now(), [0.0]) for i in ids]
        pose = np.stack([np.concatenate([w.p[0], w.q[0]]) for w in now])
        self.pose_command_w[ids] = torch.as_tensor(pose, dtype=torch.float32, device=self.device)

    def _now(self) -> float:
        return float(self._env.common_step_counter) * self._env.step_dt

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "reference_visualizer"):
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            marker_cfg.prim_path = "/Visuals/Command/reference_pose"
            self.reference_visualizer = VisualizationMarkers(marker_cfg)
            marker_cfg.prim_path = "/Visuals/Command/body_pose"
            self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
        if hasattr(self, "reference_visualizer"):
            self.reference_visualizer.set_visibility(debug_vis)
            self.body_pose_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.reference_visualizer.visualize(
            self.pose_command_w[:, :3] + self._env.scene.env_origins, self.pose_command_w[:, 3:]
        )
        body_pose_w = self.robot.data.body_com_state_w.torch[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class MpcCommandCfg(CommandTermCfg):
    """Configuration for an MPC-tracked payload reference."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the payload body the reference is for."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never mid-episode: a reference is built once, at reset."""

    solve_teacher: bool = True
    """Solve the MPC every step. Off when only a student flies: the reference is still built, the teacher idles."""

    warmup_solves: int = 20
    """Solves at each reset, all for that same instant: the MPC does one SQP iteration per solve, so the
    first plan flown is converged instead of the cold start's first guess."""

    rebuild: bool = False
    """Regenerate the MPC teacher's acados solver first; needed once per machine."""


class ApproachPoseCommand(MpcCommand):
    """A min-jerk ramp from the payload's pose to a uniformly sampled goal pose."""

    cfg: ApproachPoseCommandCfg

    def __init__(self, cfg: ApproachPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.goals = torch.zeros(self.num_envs, 7, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        r = self.cfg.ranges
        bounds = torch.tensor([r.pos_x, r.pos_y, r.pos_z, r.roll, r.pitch, r.yaw], device=self.device)
        x = sample_uniform(bounds[:, 0], bounds[:, 1], (len(ids), 6), device=self.device)
        self.goals[ids] = torch.cat([x[:, :3], quat_from_euler_xyz(x[:, 3], x[:, 4], x[:, 5])], dim=-1)
        super()._resample_command(ids)

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        goal = self.goals[env_id].cpu().numpy()
        goal = LoadState(time=start.time, p=goal[:3], q=goal[3:])
        return Approach(start=start, goal=goal, duration=self.cfg.ramp_duration, time_offset=start.time).build()


@configclass
class ApproachPoseCommandCfg(MpcCommandCfg):
    """Configuration for the approach to a random pose."""

    class_type: type = ApproachPoseCommand

    @configclass
    class Ranges:
        """Uniform ranges the goal pose is sampled from, env frame."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING

    ramp_duration: float = 3.0
    """Length of the min-jerk ramp to each sampled goal, in s."""


class FigureEightCommand(MpcCommand):
    """A figure eight, moved to start at the payload and turned by its yaw: a random `FigureEight` per reset, or a CSV's."""

    cfg: FigureEightCommandCfg

    def __init__(self, cfg: FigureEightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.figure = LoadTrajectory.from_csv(REFERENCE_DATA / cfg.csv) if cfg.csv else None

    def _random_figure(self) -> LoadTrajectory:
        r = self.cfg.ranges
        bounds = torch.tensor([r.amplitude, r.acceleration, r.yaw_rate], device=self.device)
        amplitude, acceleration, yaw_rate = sample_uniform(
            bounds[:, 0], bounds[:, 1], (3,), device=self.device
        ).tolist()
        # a FigureEight's peak acceleration at full speed is 17/8 amplitude (2 pi / period)^2
        period = 2 * np.pi * np.sqrt(17 / 8 * amplitude / acceleration)
        duration = self._env.max_episode_length_s + self.teacher.policies[0].horizon.Tf
        return FigureEight(amplitude=amplitude, period=period, yaw_rate=yaw_rate, duration=duration).build()

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        f = self.figure if self.figure is not None else self._random_figure()
        yaw = Rotation.from_quat(start.q).as_euler("ZYX")[0] - Rotation.from_quat(f.q[0]).as_euler("ZYX")[0]
        turn = Rotation.from_euler("z", yaw)
        return LoadTrajectory(
            time=f.time + start.time,
            p=start.p + turn.apply(f.p - f.p[0]),
            v=turn.apply(f.v),
            a=turn.apply(f.a),
            q=(turn * Rotation.from_quat(f.q)).as_quat(),
            w=f.w,  # body frame, so a turn of the whole path leaves the rates as they are
            alpha=f.alpha,
        )


@configclass
class FigureEightCommandCfg(MpcCommandCfg):
    """Configuration for the figure eight."""

    class_type: type = FigureEightCommand

    @configclass
    class Ranges:
        """Uniform ranges a random `FigureEight` is sampled from."""

        amplitude: tuple[float, float] = MISSING  # min max [m], half the x-span
        acceleration: tuple[float, float] = MISSING  # min max [m/s^2], the peak at full speed; sets the period
        yaw_rate: tuple[float, float] = MISSING  # min max [rad/s]

    ranges: Ranges = MISSING

    csv: str | None = None
    """A fixed figure eight in REFERENCE_DATA instead, the same at every reset. The thesis's: 40 s, at rest at
    both ends, figure_eight_v2_a2_yaw025.csv 5 x 4 m, up to 2.4 m/s and 2 m/s^2, the a4 half that size, 4.1 m/s^2."""
