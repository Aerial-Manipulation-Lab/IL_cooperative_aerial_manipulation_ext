# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher: one MPC per env, solved against the scene's robot."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from python_mpc_cusadi import DroneState, LoadState, OCPStatus, TeacherPolicy, TuningCfg
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend
from python_mpc_cusadi.ocp.problem import OcpProblem
from scipy.spatial.transform import Rotation

from ..plants import FLYCRANE_SIM, ROPE_END


class MpcTeacher:
    """One NMPC per env, owned by the pose command term. After `solve`:

    - `plan`: (num_envs, num_drones, num_nodes, 12), each drone's p, v, a, w per node, env frame.
    - `action`: (num_envs, num_drones * 12), node 1's p, v, a, w (world frame), as the action term takes it.
    - `ok`: (num_envs,) bool, False where the last solve failed.
    - `load_plan`: (num_envs, num_nodes, 7), the payload pose the last solve predicts: p, then q (XYZW).
    - `tension`: (num_envs, num_drones), the cable tensions the last solve started from, carried, never measured.
    """

    def __init__(self, env, plant=FLYCRANE_SIM, rebuild=False):
        tuning = TuningCfg()
        problem = OcpProblem.for_plant(plant, tuning)
        # only the first backend regenerates the acados artifacts; the rest reuse them
        self.policies = [
            TeacherPolicy(plant, tuning, AcadosBackend(problem, rebuild=(rebuild and i == 0)))
            for i in range(env.num_envs)
        ]
        self.plant = plant
        self.num_envs = env.num_envs
        self.num_drones = plant.num_drones
        self.num_nodes = self.policies[0].horizon.num_nodes
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
        print(f"[INFO]: MPC teacher, {self.num_envs} envs, N={self.policies[0].backend.N}")

    def reset(self, i: int, reference):
        """Cold-start env i's MPC on a new reference."""
        self.policies[i].reset()
        self.policies[i].set_reference(reference)

    def solve(self, env_ids: list[int], stime: float):
        """Solve the listed envs at time `stime` from their measured state."""
        if not env_ids:
            return
        rows = self.robot.data.body_com_state_w.torch[env_ids][:, [self.load_idx, *self.falcon_idx]]
        rows = rows.cpu().numpy().astype(float)
        origins = self.env_origins[env_ids].cpu().numpy()

        def solve(j):
            load = self._load_state(rows[j, 0], origins[j], stime)
            drones = [
                DroneState(
                    time=stime, p=r[:3] - origins[j] + Rotation.from_quat(r[3:7]).apply(ROPE_END), v=r[7:10], w=r[10:13]
                )
                for r in rows[j, 1:]
            ]
            return self.policies[env_ids[j]].solve(stime, load, drones)

        for i, (drones, load, status) in zip(env_ids, self._pool.map(solve, range(len(env_ids))), strict=True):
            self._ok[i] = status == OCPStatus.SUCCESS
            self._plan[i] = np.concatenate([drones.p, drones.v, drones.a, drones.w], axis=-1).transpose(1, 0, 2)
            self.load_plan[i] = np.concatenate([load.p, load.q], axis=-1)
            self.tension[i] = self.policies[i].warm_start.cable_prediction.t[0]
        self._publish()

    def measured_load_state(self, i: int, time: float) -> LoadState:
        """Env-frame state of env i's payload."""
        row = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        return self._load_state(row, self.env_origins[i].cpu().numpy(), time)

    @staticmethod
    def _load_state(row, origin, time) -> LoadState:
        # Isaac Lab 3.0 quaternions are XYZW like the MPC's; w goes to the payload body frame
        return LoadState(
            time=time,
            p=row[:3] - origin,
            q=row[3:7],
            v=row[7:10],
            w=Rotation.from_quat(row[3:7]).as_matrix().T @ row[10:13],
        )

    def _publish(self):
        self.plan = torch.as_tensor(self._plan, dtype=torch.float32, device=self.device)
        self.ok = torch.as_tensor(self._ok, device=self.device)
        node1 = self.plan[:, :, 1]
        self.action = node1.flatten(1)
