# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The teacher: one MPC per env, between the scene and the python_mpc_cusadi package."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import numpy as np
import torch
from python_mpc_cusadi import (
    MPC,
    AcadosSolver,
    CableTrajectory,
    DroneState,
    DroneTrajectory,
    LoadState,
    LoadTrajectory,
    Ocp,
    RigTrajectory,
    Status,
    Tuning,
)
from scipy.spatial.transform import Rotation

from ..assets import DRONE_IDX, PAYLOAD_IDX
from ..plants import FLYCRANE_SIM, ROPE_END

PAYLOAD_BODY = "load_odometry_sensor_link"
"""The body the teacher takes as the payload, which `PAYLOAD_IDX` must point at."""

DRONE_BODIES = "Falcon.*_base_link_inertia"
"""The bodies the teacher takes as the drones, which `DRONE_IDX` must point at."""


class TeacherPolicy:
    """One MPC per env, the only code that speaks both the scene's and the package's language.

    The scene is read in the world frame as torch tensors, one row per body. The package takes
    one env at a time, in the env frame, as numpy. The teacher measures, hands each env's MPC its
    measurements, and keeps what the package returns as it is, one object per env.

    Attributes:
        rig: The rig the MPCs model.
        num_drones: Drones in the rig.
        num_nodes: Nodes of the horizon, the first at now.
        node_offsets: Time of each node ahead of now, in s, of shape (num_nodes,).
        mpcs: One MPC per env, each holding its own reference.
        windows: Per env, the reference at the nodes ahead of now, as of `read_reference`.
        plans: Per env, what its last solve planned, at the nodes ahead of that solve's time.
        ok: Whether each env's last solve succeeded, of shape (num_envs,).
    """

    def __init__(self, env, rig=FLYCRANE_SIM, rebuild: bool = False):
        """Build one MPC per env on the rig.

        Args:
            env: Environment whose scene the teacher measures.
            rig: The rig the MPCs model, the sim's own unless a wrong model is being tested.
            rebuild: Whether to generate and compile the acados solver again first.
        """
        ocp = Ocp.for_rig(rig, Tuning())
        # Only the first solver generates and compiles the acados code; the rest load it.
        self.mpcs = [
            MPC(AcadosSolver(ocp, rebuild=rebuild and i == 0)) for i in range(env.num_envs)
        ]
        self.rig = rig
        self.num_drones = rig.num_drones
        self.num_nodes = ocp.tuning.num_nodes
        self.node_offsets = ocp.tuning.node_offsets
        self.robot = env.scene["robot"]
        self.env_origins = env.scene.env_origins
        self.device = env.device
        _check_bodies(self.robot)
        # Before an env's first reference or solve, it holds an empty trajectory of the right
        # length, so the terms reading it need no special case.
        self.windows = [_build_empty(self.num_drones, self.num_nodes) for _ in self.mpcs]
        self.plans = [_build_empty(self.num_drones, self.num_nodes) for _ in self.mpcs]
        self.ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        # acados releases the GIL during the C solve, so the envs' solves overlap.
        self._pool = ThreadPoolExecutor(min(env.num_envs, os.cpu_count() or 1))

    """
    Operations.
    """

    def measure_load(self, env_ids) -> list[LoadState]:
        """Read the payload of some envs, in the env frame.

        Args:
            env_ids: Envs to read.

        Returns:
            The payload of each env, with its spin in the payload's own frame as the MPC takes it.
        """
        rows = self._read_bodies(env_ids, [PAYLOAD_IDX[0]])[:, 0]
        return [
            LoadState(
                position=row[:3],
                velocity=row[7:10],
                orientation=row[3:7],
                angular_velocity=Rotation.from_quat(row[3:7]).inv().apply(row[10:13]),
            )
            for row in rows
        ]

    def set_reference(self, env_ids, references: list[LoadTrajectory]) -> None:
        """Start some envs' MPCs cold on new references.

        Args:
            env_ids: Envs to start.
            references: Their payload references, one per env.
        """
        for i, reference in zip(env_ids, references, strict=True):
            self.mpcs[i].set_reference(reference)
            self.plans[i] = _build_empty(self.num_drones, self.num_nodes)

    def read_reference(self, now: float) -> None:
        """Read every env's reference at the nodes ahead of `now`, into `windows`.

        Args:
            now: The time now, in s, on the references' clock.
        """
        loads = self.measure_load(range(len(self.mpcs)))
        for i, (mpc, load) in enumerate(zip(self.mpcs, loads, strict=True)):
            if mpc.reference is not None:
                self.windows[i] = mpc.reference.get_window(now, self.node_offsets, load.orientation)

    def solve(self, env_ids, now: float) -> None:
        """Solve some envs' MPCs at `now` from their measured state, into `plans` and `ok`.

        Args:
            env_ids: Envs to solve.
            now: The time now, in s, on the references' clock.
        """
        env_ids = list(env_ids)
        if not env_ids:
            return
        loads = self.measure_load(env_ids)
        drones = self._measure_drones(env_ids)

        def solve(j):
            return self.mpcs[env_ids[j]].step(now, loads[j], drones[j])

        statuses = []
        for i, (plan, status) in zip(
            env_ids, self._pool.map(solve, range(len(env_ids))), strict=True
        ):
            self.plans[i] = plan
            statuses.append(status == Status.SUCCESS)
        self.ok[env_ids] = torch.tensor(statuses, device=self.device)

    """
    Internal operations.
    """

    def _measure_drones(self, env_ids) -> list[list[DroneState]]:
        """Read the drones of some envs at their rope ends, in the env frame.

        The MPC plans each drone at its rope end, `ROPE_END` below the body, and reads only the
        positions.

        Args:
            env_ids: Envs to read.

        Returns:
            Per env, its drones in the rig's order.
        """
        rows = self._read_bodies(env_ids, DRONE_IDX)
        return [
            [
                DroneState(
                    position=row[:3] + Rotation.from_quat(row[3:7]).apply(ROPE_END),
                    velocity=row[7:10],
                )
                for row in env_rows
            ]
            for env_rows in rows
        ]

    def _read_bodies(self, env_ids, body_ids: list[int]) -> np.ndarray:
        """Read some bodies of some envs, in the env frame.

        Args:
            env_ids: Envs to read.
            body_ids: Bodies to read, in the articulation.

        Returns:
            Per env and body its position, orientation as the quaternion [x, y, z, w], velocity and
            angular velocity along the world axes, of shape (len(env_ids), len(body_ids), 13).
        """
        env_ids = list(env_ids)
        rows = self.robot.data.body_com_state_w.torch[env_ids][:, body_ids]
        rows = rows.cpu().numpy().astype(float)
        rows[..., :3] -= self.env_origins[env_ids].cpu().numpy()[:, None]
        return rows


"""
Helpers.
"""


def _build_empty(num_drones: int, num_nodes: int) -> RigTrajectory:
    """Build a rig trajectory of zeros, with the payload upright, over the horizon's nodes.

    Args:
        num_drones: Drones in the rig.
        num_nodes: Nodes of the horizon.

    Returns:
        The trajectory, every member zero but the payload's orientation, the identity.
    """

    def empty(cls):
        members = {}
        for f in fields(cls):
            if f.name != "time":
                width = f.metadata["width"]
                members[f.name] = np.zeros((num_nodes, width) if width > 1 else num_nodes)
        return cls(time=np.zeros(num_nodes), **members)

    load = empty(LoadTrajectory)
    load.orientation[:, 3] = 1.0
    return RigTrajectory(
        load=load,
        cables=[empty(CableTrajectory) for _ in range(num_drones)],
        drones=[empty(DroneTrajectory) for _ in range(num_drones)],
    )


def _check_bodies(robot) -> None:
    """Check the asset's body indices against the body names, since every term relies on them.

    Args:
        robot: The flycrane articulation.

    Raises:
        ValueError: If the indices do not point at the payload and drone bodies.
    """
    for names, expected in ((PAYLOAD_BODY, PAYLOAD_IDX), (DRONE_BODIES, DRONE_IDX)):
        found = robot.find_bodies(names)[0]
        if list(found) != list(expected):
            raise ValueError(
                f"{names!r} are bodies {list(found)}, not {list(expected)} as "
                "IL_mav_carry_ext.assets says"
            )
