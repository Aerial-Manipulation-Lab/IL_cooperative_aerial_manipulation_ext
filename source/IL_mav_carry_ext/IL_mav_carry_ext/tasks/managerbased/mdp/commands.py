# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
    configclass,  # Imported explicitly, since isaaclab.utils lazy-exports this name.
)
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, sample_uniform
from python_mpc_cusadi import Approach, LoadState, LoadTrajectory, PlantCfg, Shape, SplinePath
from scipy.spatial.transform import Rotation

from ....mpc.teacher import MpcTeacher
from ....plants import FLYCRANE_SIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MpcCommand(CommandTerm):
    """A reference for the payload in every env, and the MPC teacher that tracks it.

    At every reset each env gets a new reference, built from where the payload is, and hands it
    to its own MPC. The teacher then solves once per step, at the time the references are stamped
    with. Subclasses decide which reference to build. The command itself is the reference pose
    now, position then quaternion (x, y, z, qx, qy, qz, qw) in the env frame.
    """

    cfg: MpcCommandCfg

    def __init__(self, cfg: MpcCommandCfg, env: ManagerBasedRLEnv):
        """Set up the teacher, one MPC per env, with no references yet.

        Args:
            cfg: Configuration of the command.
            env: Environment the command runs in.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 6] = 1.0
        # These hold the payload's tracking RMSE over the episode so far, which Isaac Lab hands
        # out when the episode ends.
        self.metrics["pos_rmse_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["att_rmse_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.references: list = [None] * self.num_envs
        self.drone_references: list = [None] * self.num_envs
        self.teacher = MpcTeacher(env, plant=cfg.model or FLYCRANE_SIM, rebuild=cfg.rebuild)
        self.solved_at = np.full(self.num_envs, -1)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """The reference pose now, as position then quaternion, in the env frame.

        Returns:
            Pose (x, y, z, qx, qy, qz, qw), of shape (num_envs, 7).
        """
        return self.pose_command_w

    """
    Operations.
    """

    def get_reference(self, env_id: int) -> LoadTrajectory:
        """Get one env's reference, as the MPC tracks it.

        Args:
            env_id: Index of the env.

        Returns:
            The env's reference, or None before its first reset.
        """
        return self.references[int(env_id)]

    """
    Internal operations.
    """

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        """Build one env's reference, leaving the payload's pose at rest now.

        Args:
            env_id: Index of the env.
            start: The payload's pose now, at rest, and the time now.

        Returns:
            The reference, starting at `start.time`.
        """
        raise NotImplementedError

    def _compute_drone_reference(self, env_id: int) -> np.ndarray:
        """Compute where each drone should be if the payload followed its reference exactly.

        The teacher's own cable reference, which follows from the payload reference alone, is
        placed on the payload reference. This gives each drone's rope end in the env frame.

        Args:
            env_id: Index of the env.

        Returns:
            Per reference sample and drone, the position, velocity and acceleration of the rope
            end and the cable tension, of shape (num_samples, num_drones, 10).
        """
        policy = self.teacher.policies[env_id]
        ref, cables = policy.load_reference, policy.cable_reference
        lengths = np.array([d.cable_length for d in policy.plant.drones])
        # The cable reference has no angular acceleration, so it is differenced from the rate.
        cable_acc = np.gradient(cables.r, ref.time, axis=0)
        p, v, a = policy.problem.physics.compute_drone_states(
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
        """Give the reset envs a new reference and start their teacher on it.

        Args:
            env_ids: Envs being reset.
        """
        ids = torch.arange(self.num_envs, device=self.device)[env_ids].tolist()
        measured = self.robot.data.body_com_state_w.torch[ids, self.body_idx].cpu().numpy()
        origins = self._env.scene.env_origins[ids].cpu().numpy()
        for j, i in enumerate(ids):
            start = LoadState(
                time=self._get_time(), p=measured[j, :3] - origins[j], q=measured[j, 3:7]
            )
            self.references[i] = self._build_reference(i, start)
            self.teacher.reset(i, self.references[i])
            self.drone_references[i] = self._compute_drone_reference(i)
        # The MPC does one optimisation step per solve, so several solves at the same instant
        # make the first plan flown a converged one instead of the cold start's first guess.
        for _ in range(self.cfg.warmup_solves if self.cfg.solve_teacher else 0):
            self._solve(ids)
        self._update_pose_command(ids)

    def _update_command(self):
        """Solve the teacher for every env not solved yet this step, and update the pose command."""
        if self.cfg.solve_teacher:
            self._solve(np.flatnonzero(self.solved_at != self._env.common_step_counter).tolist())
        self._update_pose_command(range(self.num_envs))

    def _update_metrics(self):
        """Add this step's tracking error to each episode's running RMSE."""
        # The metrics update before the command does, so the pose command is first brought to
        # this step's time.
        self._update_pose_command(range(self.num_envs))
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_com_state_w.torch[:, self.body_idx, :3]
            - self._env.scene.env_origins,
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
        """Solve the teacher for some envs at the time now.

        Args:
            env_ids: Envs to solve.
        """
        self.teacher.solve(env_ids, self._get_time())
        self.solved_at[env_ids] = self._env.common_step_counter

    def _update_pose_command(self, env_ids):
        """Set the pose command of some envs to their reference pose now.

        Args:
            env_ids: Envs to update.
        """
        ids = list(env_ids)
        now = [self.references[i].get_window(self._get_time(), [0.0]) for i in ids]
        pose = np.stack([np.concatenate([w.p[0], w.q[0]]) for w in now])
        self.pose_command_w[ids] = torch.as_tensor(pose, dtype=torch.float32, device=self.device)

    def _get_time(self) -> float:
        """Get the time now, the clock the references are stamped with, in s.

        Returns:
            Steps taken times the step time.
        """
        return float(self._env.common_step_counter) * self._env.step_dt

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Show or hide the reference pose and the payload pose in the viewer.

        Args:
            debug_vis: Whether to show them.
        """
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
        """Draw the reference pose and the payload pose for the frame being rendered.

        Args:
            event: Render event from Isaac Lab, unused.
        """
        if not self.robot.is_initialized:
            return
        self.reference_visualizer.visualize(
            self.pose_command_w[:, :3] + self._env.scene.env_origins, self.pose_command_w[:, 3:]
        )
        body_pose_w = self.robot.data.body_com_state_w.torch[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class MpcCommandCfg(CommandTermCfg):
    """Configuration of a payload reference tracked by the MPC teacher."""

    # the payload
    asset_name: str = MISSING  # asset the reference is for
    body_name: str = MISSING  # payload body in that asset

    # timing
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # never mid-episode, only at reset

    # teacher
    solve_teacher: bool = True  # off when only a student flies, the reference is still built
    warmup_solves: int = 20  # solves at each reset, all for that same instant
    # The plant the teacher and every model-based input assume, None for the sim's own. Set it
    # apart from the sim to test a wrong model (validate.py --model_cable_length).
    model: PlantCfg | None = None
    rebuild: bool = False  # regenerate the teacher's acados solver first, needed once per machine


class ApproachPoseCommand(MpcCommand):
    """A smooth move of the payload from where it is to a random goal pose, then holding it."""

    cfg: ApproachPoseCommandCfg

    def __init__(self, cfg: ApproachPoseCommandCfg, env: ManagerBasedRLEnv):
        """Set up the command with no goals yet.

        Args:
            cfg: Configuration of the command.
            env: Environment the command runs in.
        """
        super().__init__(cfg, env)
        self.goals = torch.zeros(self.num_envs, 7, device=self.device)

    """
    Internal operations.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Draw a new goal pose for the reset envs, then build their references.

        Args:
            env_ids: Envs being reset.
        """
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        r = self.cfg.ranges
        bounds = torch.tensor(
            [r.pos_x, r.pos_y, r.pos_z, r.roll, r.pitch, r.yaw], device=self.device
        )
        x = sample_uniform(bounds[:, 0], bounds[:, 1], (len(ids), 6), device=self.device)
        self.goals[ids] = torch.cat(
            [x[:, :3], quat_from_euler_xyz(x[:, 3], x[:, 4], x[:, 5])], dim=-1
        )
        super()._resample_command(ids)

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        """Build a minimum-jerk move from the payload's pose to the env's goal.

        Args:
            env_id: Index of the env.
            start: The payload's pose now, at rest, and the time now.

        Returns:
            The move, starting at `start.time`.
        """
        goal = self.goals[env_id].cpu().numpy()
        goal = LoadState(time=start.time, p=goal[:3], q=goal[3:])
        return Approach(
            start=start,
            goal=goal,
            duration=self.cfg.ramp_duration,
            dt=self._env.step_dt,
            time_offset=start.time,
        ).build()


@configclass
class ApproachPoseCommandCfg(MpcCommandCfg):
    """Configuration of the move to a random goal pose."""

    class_type: type = ApproachPoseCommand

    @configclass
    class Ranges:
        """Uniform ranges the goal pose is drawn from, in the env frame."""

        # ranges, as (min, max)
        pos_x: tuple[float, float] = MISSING  # m
        pos_y: tuple[float, float] = MISSING  # m
        pos_z: tuple[float, float] = MISSING  # m
        roll: tuple[float, float] = MISSING  # rad
        pitch: tuple[float, float] = MISSING  # rad
        yaw: tuple[float, float] = MISSING  # rad

    # goal
    ranges: Ranges = MISSING
    ramp_duration: float = 3.0  # length of the move to each goal, s


class ShapeTrajectoryCommand(MpcCommand):
    """A shape for the payload to fly, moved to start at the payload and turned to its heading.

    Each env flies one entry of the configured shapes, env i the entry i modulo their number. An
    entry is either a fixed `Shape`, the same at every reset, or a random one (`RandomFigureEight`,
    `RandomWalk`) drawn anew at every reset.
    """

    cfg: ShapeTrajectoryCommandCfg

    """
    Internal operations.
    """

    def _get_shape(self, env_id: int, start: LoadState) -> Shape | SplinePath:
        """Get the shape one env flies, drawing a random one anew.

        Args:
            env_id: Index of the env.
            start: The payload's pose now, which a random walk needs for its floor.

        Returns:
            The shape, still around the origin and at heading 0.
        """
        entry = self.cfg.shapes[env_id % len(self.cfg.shapes)]
        return (
            entry if isinstance(entry, Shape) else entry.draw(start, self._env.max_episode_length_s)
        )

    def _build_reference(self, env_id: int, start: LoadState) -> LoadTrajectory:
        """Build the env's shape as a reference, starting at the payload and turned to its heading.

        Args:
            env_id: Index of the env.
            start: The payload's pose now, at rest, and the time now.

        Returns:
            The reference, starting at `start.time`.
        """
        # The reference runs past the episode by one MPC horizon, so the last plans still have a
        # reference, and has one sample per step, so the MPC never aims a whole sample ahead.
        duration = self._env.max_episode_length_s + self.teacher.policies[0].horizon.Tf
        f = replace(self._get_shape(env_id, start), duration=duration, dt=self._env.step_dt).build()
        yaw = (
            Rotation.from_quat(start.q).as_euler("ZYX")[0]
            - Rotation.from_quat(f.q[0]).as_euler("ZYX")[0]
        )
        turn = Rotation.from_euler("z", yaw)
        return LoadTrajectory(
            time=f.time + start.time,
            p=start.p + turn.apply(f.p - f.p[0]),
            v=turn.apply(f.v),
            a=turn.apply(f.a),
            q=(turn * Rotation.from_quat(f.q)).as_quat(),
            # The rates are in the body frame, so turning the whole path leaves them as they are.
            w=f.w,
            alpha=f.alpha,
        )


@configclass
class ShapeTrajectoryCommandCfg(MpcCommandCfg):
    """Configuration of the shape the payload flies."""

    class_type: type = ShapeTrajectoryCommand

    # shapes
    # Env i flies shapes[i % len(shapes)], a fixed `Shape` or a `RandomFigureEight` or
    # `RandomWalk`. Their duration is set to the episode's.
    shapes: tuple = MISSING
