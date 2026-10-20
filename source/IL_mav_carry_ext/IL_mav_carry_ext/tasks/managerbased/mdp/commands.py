# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms: a payload reference per env, and the NMPC teacher tracking it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import (
    configclass,  # explicit: isaaclab.utils lazy-exports this name and it can be shadowed by the submodule
)
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, sample_uniform
from python_mpc_cusadi import Approach, FigureEight, LoadState, LoadTrajectory, PlantCfg, Shape, SplinePath
from python_mpc_cusadi.signals.specs import SPLINE_DEGREE
from scipy.spatial.transform import Rotation

from ....mpc.teacher import MpcTeacher
from ....plants import FLYCRANE_SIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        # the payload's tracking RMSE over the episode so far; Isaac Lab hands them out when it ends
        self.metrics["pos_rmse_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["att_rmse_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.references: list = [None] * self.num_envs
        self.drone_references: list = [None] * self.num_envs
        self.teacher = MpcTeacher(env, plant=cfg.model or FLYCRANE_SIM, rebuild=cfg.rebuild)
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

    def _drone_reference(self, env_id: int) -> np.ndarray:
        """Where each drone should be if the payload flew its reference exactly: the teacher's own cable reference
        (from the reference alone) placed on the reference, rope ends, env frame. Per reference sample and drone p, v,
        a and the cable tension: (num_samples, num_drones, 10)."""
        policy = self.teacher.policies[env_id]
        ref, cables = policy.traj, policy.cable_traj
        lengths = np.array([d.cable_length for d in policy.plant.drones])
        # the cable reference carries no angular acceleration; difference its rate
        cable_acc = np.gradient(cables.r, ref.time, axis=0)
        p, v, a = policy.problem.physics.calculate_drone_states(
            ref.p,
            ref.v,
            Rotation.from_quat(ref.q).as_matrix(),
            ref.w,
            ref.a,
            ref.alpha,
            cables.s,
            cables.r,
            cable_acc,
            lengths,
        )
        return np.concatenate([p, v, a, cables.t[..., None]], axis=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids].tolist()
        measured = self.robot.data.body_com_state_w.torch[ids, self.body_idx].cpu().numpy()
        origins = self._env.scene.env_origins[ids].cpu().numpy()
        for j, i in enumerate(ids):
            start = LoadState(time=self._now(), p=measured[j, :3] - origins[j], q=measured[j, 3:7])
            self.references[i] = self._build_reference(i, start)
            self.teacher.reset(i, self.references[i])
            self.drone_references[i] = self._drone_reference(i)
        for _ in range(self.cfg.warmup_solves if self.cfg.solve_teacher else 0):
            self._solve(ids)
        self._track(ids)

    def _update_command(self):
        if self.cfg.solve_teacher:
            self._solve(np.flatnonzero(self.solved_at != self._env.common_step_counter).tolist())
        self._track(range(self.num_envs))

    def _update_metrics(self):
        """Add this step's error to each episode's running RMSE; Isaac Lab zeroes them at each reset."""
        # metrics update before the command does, so look up the reference at this step's time first
        self._track(range(self.num_envs))
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_com_state_w.torch[:, self.body_idx, :3] - self._env.scene.env_origins,
            self.robot.data.body_com_state_w.torch[:, self.body_idx, 3:7],
        )
        n = self._env.episode_length_buf.clamp(min=1)
        for name, error in (
            ("pos_rmse_m", pos_error.norm(dim=-1)),
            ("att_rmse_deg", torch.rad2deg(rot_error.norm(dim=-1))),
        ):
            rmse = self.metrics[name]
            rmse[:] = (rmse**2 + (error**2 - rmse**2) / n).sqrt()

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

    model: PlantCfg | None = None
    """The plant the teacher and every model-based input assume; None: the sim's own, `FLYCRANE_SIM`. Set it apart
    from the sim to test a wrong model (validate.py --model_cable_length, --model_attach_offset)."""

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
        return Approach(
            start=start, goal=goal, duration=self.cfg.ramp_duration, dt=self._env.step_dt, time_offset=start.time
        ).build()


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


class ShapeTrajectoryCommand(MpcCommand):
    """Each env's shape, moved to start at the payload and turned by its yaw: env i flies `cfg.shapes[i % len]`, a
    fixed `Shape`, or a `RandomFigureEight` or `RandomWalk` drawn anew at every reset."""

    cfg: ShapeTrajectoryCommandCfg

    def _shape(self, env_id: int, start: LoadState) -> Shape | SplinePath:
        """Env `env_id`'s entry, round robin; a random one drawn anew."""
        entry = self.cfg.shapes[env_id % len(self.cfg.shapes)]
        return entry if isinstance(entry, Shape) else entry.draw(start, self._env.max_episode_length_s)

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        # long enough for the MPC's horizon at the episode's end, a sample per step: coarser, the MPC and the
        # student both read a sample up to a whole spacing ahead of each node
        duration = self._env.max_episode_length_s + self.teacher.policies[0].horizon.Tf
        f = replace(self._shape(env_id, start), duration=duration, dt=self._env.step_dt).build()
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
class ShapeTrajectoryCommandCfg(MpcCommandCfg):
    """Configuration for the shape trajectory."""

    class_type: type = ShapeTrajectoryCommand

    shapes: tuple = MISSING
    """Env i flies shapes[i % len(shapes)]: a fixed `Shape`, the same at every reset, or a `RandomFigureEight` or
    `RandomWalk`, drawn anew at every reset. Their duration is set to the episode's."""


@configclass
class RandomFigureEight:
    """A random `FigureEight` per reset, from uniform ranges."""

    amplitude: tuple[float, float] = MISSING  # min max [m], half the x-span
    acceleration: tuple[float, float] = MISSING  # min max [m/s^2], the peak at full speed; sets the period
    yaw_rate: tuple[float, float] = MISSING  # min max [rad/s]

    def draw(self, start: LoadState, episode: float) -> FigureEight:
        amplitude, acceleration, yaw_rate = (
            np.random.uniform(*bounds) for bounds in (self.amplitude, self.acceleration, self.yaw_rate)
        )
        # a FigureEight's peak acceleration at full speed is 17/8 amplitude (2 pi / period)^2
        period = 2 * np.pi * np.sqrt(17 / 8 * amplitude / acceleration)
        return FigureEight(amplitude=amplitude, period=period, yaw_rate=yaw_rate)


WALK_INTERVAL = 0.95
"""Time between a random walk's control points, per sqrt(size / acceleration): calibrated offline so the median
peak acceleration over the walks is the sampled one."""

WALK_STEP = 1.22
"""A random walk's longest step, per size: calibrated offline so the median peak speed is sqrt(size acceleration)."""


@configclass
class RandomWalk:
    """A random walk per reset, flown from rest to rest within the episode as a `SplinePath`.

    Each draw takes the room the walk's turns take (size, v^2 / a), its peak acceleration and its peak heading
    rate from uniform ranges. Size sets the step between control points, size and acceleration the time
    between them, so the reference peaks at about the speed and acceleration drawn (`WALK_INTERVAL`,
    `WALK_STEP`). The course keeps its direction with a random turn per step, mostly level, and turns back off
    the walls: within `reach` of its start sideways, `vertical` of its start height and above `floor`. The
    spline stays inside its control points, so inside those walls too. The heading rate takes a persistent
    random walk within its peak, which the spline keeps to exactly.

    It is drawn around the origin at heading 0 and placed like any shape: moved to the payload and turned by
    its yaw.
    """

    size: tuple[float, float] = MISSING  # min max [m], v^2 / a: the room its turns take
    acceleration: tuple[float, float] = MISSING  # min max [m/s^2], about its peak
    yaw_rate: tuple[float, float] = MISSING  # min max [rad/s], its peak heading rate, either way round

    reach: float = 4.0
    """How far from its start the walk may go along x and y, m, before it is turned."""

    vertical: float = 0.6
    """How far above or below its start height the walk may go, m."""

    floor: float = 0.4
    """The lowest the walk may go, m."""

    rest: float = 1.0
    """Time at rest at each end of the episode, s."""

    def draw(self, start: LoadState, episode: float) -> SplinePath:
        size, acceleration, yaw_rate = (
            np.random.uniform(*bounds) for bounds in (self.size, self.acceleration, self.yaw_rate)
        )
        interval, step = WALK_INTERVAL * np.sqrt(size / acceleration), WALK_STEP * size
        # as many control points as come to rest `rest` before the episode ends
        n = max(int((episode - 2 * self.rest) / interval) - SPLINE_DEGREE + 2, 2)

        # the walls around its start, the origin here; the floor needs the start's own height
        lo = np.array([-self.reach, -self.reach, max(-self.vertical, self.floor - start.p[2])])
        hi = np.array([self.reach, self.reach, self.vertical])
        level = np.array([1.0, 1.0, 0.3])  # mostly level: the height band is narrow
        course = np.random.normal(size=3) * level
        course /= np.linalg.norm(course)
        points = [np.zeros(3)]
        for _ in range(n - 1):
            course += np.random.normal(size=3) * level
            course /= np.linalg.norm(course)
            p = points[-1] + step * np.random.uniform(0.5, 1.0) * course
            course[(p < lo) | (p > hi)] *= -1  # turn back off the walls
            points.append(np.clip(p, lo, hi))

        rate, yaws = 0.0, [0.0]
        for _ in range(n - 1):
            rate = np.clip(rate + 0.3 * yaw_rate * np.random.normal(), -yaw_rate, yaw_rate)
            yaws.append(yaws[-1] + rate * interval)

        return SplinePath(points=np.array(points), yaws=np.array(yaws), interval=interval, start=self.rest)
