# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The teacher: one MPC per env, solved against the scene's robot."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from python_mpc_cusadi import MPC, AcadosSolver, DroneState, LoadState, Ocp, Status, Tuning
from scipy.spatial.transform import Rotation

from ..plants import FLYCRANE_SIM, ROPE_END


class TeacherPolicy:
    """One MPC per env, owned by the pose command term. After `solve`:

    - `plan`: (num_envs, num_drones, num_nodes, 12), each drone's p, v, a, w per node, env frame; the nodes
      are `node_offsets_s` ahead of now.
    - `ok`: (num_envs,) bool, False where the last solve failed.
    - `load_plan`: (num_envs, num_nodes, 7), the payload pose the last solve predicts: p, then q (XYZW).
    - `tension`: (num_envs, num_drones), the cable tensions at node 0 of the last plan, estimated, never measured.
    """

    def __init__(self, env, rig=FLYCRANE_SIM, rebuild=False):
        ocp = Ocp.for_rig(rig, Tuning())
        # only the first solver regenerates the acados artifacts; the rest reuse them
        self.policies = [
            MPC(AcadosSolver(ocp, rebuild=(rebuild and i == 0))) for i in range(env.num_envs)
        ]
        self.rig = rig
        self.num_envs = env.num_envs
        self.num_drones = rig.num_drones
        self.num_nodes = ocp.tuning.num_nodes
        self.node_offsets_s = ocp.tuning.node_offsets
        self.robot = env.scene["robot"]
        self.load_idx = self.robot.find_bodies("load_odometry_sensor_link")[0][0]
        self.falcon_idx = self.robot.find_bodies("Falcon.*_base_link_inertia")[0]
        self.env_origins = env.scene.env_origins
        self.device = env.device

        self._plan = np.zeros((self.num_envs, self.num_drones, self.num_nodes, 12))
        self._ok = np.ones(self.num_envs, dtype=bool)
        self.load_plan = np.zeros((self.num_envs, self.num_nodes, 7))
        self.tension = np.zeros((self.num_envs, self.num_drones))
        self._publish()
        # acados releases the GIL during the C solve, so the envs' solves overlap
        self._pool = ThreadPoolExecutor(min(self.num_envs, os.cpu_count() or 1))

    def reset(self, i: int, reference):
        """Cold-start env i's MPC on a new reference."""
        self.policies[i].set_reference(reference)

    def solve(self, env_ids: list[int], stime: float):
        """Solve the listed envs at time `stime` from their measured state."""
        if not env_ids:
            return
        rows = self.robot.data.body_com_state_w.torch[env_ids][:, [self.load_idx, *self.falcon_idx]]
        rows = rows.cpu().numpy().astype(float)
        origins = self.env_origins[env_ids].cpu().numpy()

        def solve(j):
            load = self._load_state(rows[j, 0], origins[j])
            # The MPC plans each drone at its rope end, and reads only the positions.
            drones = [
                DroneState(
                    position=r[:3] - origins[j] + Rotation.from_quat(r[3:7]).apply(ROPE_END),
                    velocity=r[7:10],
                )
                for r in rows[j, 1:]
            ]
            return self.policies[env_ids[j]].step(stime, load, drones)

        for i, (plan, status) in zip(
            env_ids, self._pool.map(solve, range(len(env_ids))), strict=True
        ):
            self._ok[i] = status == Status.SUCCESS
            self._plan[i] = np.stack(
                [
                    np.concatenate(
                        [d.position, d.velocity, d.acceleration, d.angular_velocity], axis=-1
                    )
                    for d in plan.drones
                ]
            )
            self.load_plan[i] = np.concatenate([plan.load.position, plan.load.orientation], -1)
            self.tension[i] = [c.tension[0] for c in plan.cables]
        self._publish()

    def measured_load_state(self, i: int) -> LoadState:
        """Env-frame state of env i's payload."""
        row = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        return self._load_state(row, self.env_origins[i].cpu().numpy())

    @staticmethod
    def _load_state(row, origin) -> LoadState:
        # Isaac Lab 3.0 quaternions are XYZW like the MPC's; the spin goes to the payload body frame
        return LoadState(
            position=row[:3] - origin,
            orientation=row[3:7],
            velocity=row[7:10],
            angular_velocity=Rotation.from_quat(row[3:7]).as_matrix().T @ row[10:13],
        )

    def _publish(self):
        self.plan = torch.as_tensor(self._plan, dtype=torch.float32, device=self.device)
        self.ok = torch.as_tensor(self._ok, device=self.device)
