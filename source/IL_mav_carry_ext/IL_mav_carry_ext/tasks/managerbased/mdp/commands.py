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
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import (
    configclass,  # Imported explicitly, since isaaclab.utils lazy-exports this name.
)
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, sample_uniform
from python_mpc_cusadi import Approach, LoadState, LoadTrajectory, Rig, Shape, SplinePath
from scipy.spatial.transform import Rotation

from ....assets import PAYLOAD_IDX
from ....mpc.teacher import TeacherPolicy
from ....plants import FLYCRANE_SIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReferenceCommand(CommandTerm):
    """A reference for the payload in every env, and the MPC teacher that tracks it.

    At every reset each env gets a new reference, built from where its payload is, and its MPC
    starts cold on it. The teacher then solves once per step, at the time the references are
    stamped with. Subclasses decide which reference to build. The command itself is the reference
    pose now, position then quaternion (x, y, z, qx, qy, qz, qw) in the env frame.
    """

    cfg: ReferenceCommandCfg

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedRLEnv):
        """Set up the teacher, one MPC per env, with no references yet.

        Args:
            cfg: Configuration of the command.
            env: Environment the command runs in.
        """
        super().__init__(cfg, env)
        self.teacher = TeacherPolicy(env, rig=cfg.model or FLYCRANE_SIM, rebuild=cfg.rebuild)
        self.solved_at = np.full(self.num_envs, -1.0)
        # These hold the payload's tracking RMSE over the episode so far, which Isaac Lab hands
        # out when the episode ends.
        self.metrics["pos_rmse_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["att_rmse_deg"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """The reference pose now, as position then quaternion, in the env frame.

        Returns:
            Pose (x, y, z, qx, qy, qz, qw), of shape (num_envs, 7).
        """
        pose = [
            np.concatenate([w.load.position[0], w.load.orientation[0]])
            for w in self.teacher.windows
        ]
        return torch.as_tensor(np.stack(pose), dtype=torch.float32, device=self.device)

    @property
    def now(self) -> float:
        """The time now, in s, the clock the references are stamped with.

        Returns:
            Steps taken times the step time.
        """
        return float(self._env.common_step_counter) * self._env.step_dt

    """
    Internal operations.
    """

    def _build_reference(self, env_id: int, start: LoadState, now: float) -> LoadTrajectory:
        """Build one env's reference, leaving the payload's pose at rest now.

        Args:
            env_id: Index of the env.
            start: The payload as measured now, at rest.
            now: The time now, in s, which the reference starts at.

        Returns:
            The reference, starting at `now`.
        """
        raise NotImplementedError

    def _resample_command(self, env_ids: Sequence[int]):
        """Give the reset envs a new reference and start their teacher on it.

        Args:
            env_ids: Envs being reset.
        """
        ids = torch.arange(self.num_envs, device=self.device)[env_ids].tolist()
        now = self.now
        starts = self.teacher.measure_load(ids)
        references = [
            self._build_reference(i, start, now) for i, start in zip(ids, starts, strict=True)
        ]
        self.teacher.set_reference(ids, references)
        self.teacher.read_reference(now)
        # The MPC does one optimisation step per solve, so several solves at the same instant
        # make the first plan flown a converged one instead of the cold start's first guess.
        if self.cfg.solve_teacher:
            for _ in range(self.cfg.warmup_solves):
                self._solve(ids)

    def _update_command(self):
        """Solve the teacher for every env not solved yet this step."""
        if self.cfg.solve_teacher:
            self._solve(np.flatnonzero(self.solved_at != self.now).tolist())

    def _update_metrics(self):
        """Add this step's tracking error to each episode's running RMSE."""
        # Isaac Lab updates the metrics before the command, so the reference is brought to this
        # step's time first.
        self.teacher.read_reference(self.now)
        pose = self.command
        payload = self.teacher.robot.data.body_com_state_w.torch[:, PAYLOAD_IDX[0]]
        pos_error, rot_error = compute_pose_error(
            pose[:, :3], pose[:, 3:], payload[:, :3] - self._env.scene.env_origins, payload[:, 3:7]
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
        self.teacher.solve(env_ids, self.now)
        self.solved_at[env_ids] = self.now

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
        if not self.teacher.robot.is_initialized:
            return
        pose = self.command
        self.reference_visualizer.visualize(pose[:, :3] + self._env.scene.env_origins, pose[:, 3:])
        payload = self.teacher.robot.data.body_com_state_w.torch[:, PAYLOAD_IDX[0]]
        self.body_pose_visualizer.visualize(payload[:, :3], payload[:, 3:7])


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    """Configuration of a payload reference tracked by the MPC teacher."""

    # timing
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # never mid-episode, only at reset

    # teacher
    solve_teacher: bool = True  # off when only a student flies, the reference is still built
    warmup_solves: int = 20  # solves at each reset, all for that same instant
    # The rig the teacher and every model-based input assume, None for the sim's own. Set it
    # apart from the sim to test a wrong model (validate.py --model_cable_length).
    model: Rig | None = None
    rebuild: bool = False  # regenerate the teacher's acados solver first, needed once per machine


class ApproachPoseCommand(ReferenceCommand):
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

    def _build_reference(self, env_id: int, start: LoadState, now: float) -> LoadTrajectory:
        """Build a minimum-jerk move from the payload's pose to the env's goal.

        Args:
            env_id: Index of the env.
            start: The payload as measured now, at rest.
            now: The time now, in s, which the move starts at.

        Returns:
            The move, starting at `now`.
        """
        goal = self.goals[env_id].cpu().numpy()
        goal = LoadState(position=goal[:3], orientation=goal[3:])
        return Approach(
            start=start,
            goal=goal,
            duration=self.cfg.ramp_duration,
            dt=self._env.step_dt,
            time_offset=now,
        ).build()


@configclass
class ApproachPoseCommandCfg(ReferenceCommandCfg):
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


class ShapeTrajectoryCommand(ReferenceCommand):
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
            start: The payload as measured now, which a random walk needs for its floor.

        Returns:
            The shape, still around the origin and at heading 0.
        """
        entry = self.cfg.shapes[env_id % len(self.cfg.shapes)]
        return (
            entry if isinstance(entry, Shape) else entry.draw(start, self._env.max_episode_length_s)
        )

    def _build_reference(self, env_id: int, start: LoadState, now: float) -> LoadTrajectory:
        """Build the env's shape as a reference, starting at the payload and turned to its heading.

        Args:
            env_id: Index of the env.
            start: The payload as measured now, at rest.
            now: The time now, in s, which the reference starts at.

        Returns:
            The reference, starting at `now`.
        """
        # The reference runs past the episode by one MPC horizon, so the last plans still have a
        # reference, and has one sample per step, so the MPC never aims a whole sample ahead.
        duration = self._env.max_episode_length_s + self.teacher.node_offsets[-1]
        shape = replace(self._get_shape(env_id, start), duration=duration, dt=self._env.step_dt)
        return _place_at(shape.build(), start, now)


@configclass
class ShapeTrajectoryCommandCfg(ReferenceCommandCfg):
    """Configuration of the shape the payload flies."""

    class_type: type = ShapeTrajectoryCommand

    # shapes
    # Env i flies shapes[i % len(shapes)], a fixed `Shape` or a `RandomFigureEight` or
    # `RandomWalk`. Their duration is set to the episode's.
    shapes: tuple = MISSING


"""
Helpers.
"""


def _place_at(shape: LoadTrajectory, start: LoadState, now: float) -> LoadTrajectory:
    """Move a shape built around the origin at heading 0 to the payload, turned to its heading.

    Args:
        shape: The shape, starting at time 0.
        start: The payload as measured now, whose position and heading the shape starts from.
        now: The time now, in s, which the shape starts at.

    Returns:
        The shape as a reference in the env frame, starting at `now`.
    """
    yaw = (
        Rotation.from_quat(start.orientation).as_euler("ZYX")[0]
        - Rotation.from_quat(shape.orientation[0]).as_euler("ZYX")[0]
    )
    turn = Rotation.from_euler("z", yaw)
    return LoadTrajectory(
        time=shape.time + now,
        position=start.position + turn.apply(shape.position - shape.position[0]),
        velocity=turn.apply(shape.velocity),
        acceleration=turn.apply(shape.acceleration),
        orientation=(turn * Rotation.from_quat(shape.orientation)).as_quat(),
        # The rates are in the body frame, so turning the whole path leaves them as they are.
        angular_velocity=shape.angular_velocity,
        angular_acceleration=shape.angular_acceleration,
    )
