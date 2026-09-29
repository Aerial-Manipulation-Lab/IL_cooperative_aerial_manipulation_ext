# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher that computes per-env drone setpoints from Isaac Lab state."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from python_mpc_cusadi import DroneState, LoadState, OCPStatus, TeacherPolicy, TuningCfg
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend
from python_mpc_cusadi.ocp.problem import OcpProblem
from scipy.spatial.transform import Rotation

from ..plants import FLYCRANE_SIM


class MpcTeacher:
    """One NMPC per env, solving against the scene's robot every env step.

    Owns the acados backends and the bookkeeping of which envs' last solve
    failed; references come from the task's command term, which owns their
    lifecycle. The env is only read from, never modified: resetting done envs
    stays the wrapper's (or the caller's) job.
    """

    COMMAND_NAME = "pose_command"
    """Command term the ramped references are read from."""

    def __init__(self, env, plant=FLYCRANE_SIM, tuning=None, rebuild=False, verbose=True, solve_threads=None):
        base = env.unwrapped
        self.env = base
        self.plant = plant
        self.verbose = verbose

        tuning = tuning if tuning is not None else TuningCfg()
        print(f"[INFO]: compiling the shared MPC problem once for {base.num_envs} envs...")
        problem = OcpProblem.for_plant(plant, tuning)
        # only the first backend regenerates the acados artifacts; the rest just
        # open another solver handle on the same compiled problem (cheap).
        self.policies = [
            TeacherPolicy(plant, tuning, AcadosBackend(problem, rebuild=(rebuild and i == 0)))
            for i in range(base.num_envs)
        ]

        self.num_envs = base.num_envs
        self.num_drones = self.policies[0].num_drones
        self.step_dt = base.step_dt
        self.robot = base.scene["robot"]
        self.load_idx = self.robot.find_bodies("load_odometry_sensor_link")[0][0]
        # same bodies the low-level controller reads the drones from
        self.falcon_idx = self.robot.find_bodies("Falcon.*_base_link_inertia")[0]
        self.env_origins = base.scene.env_origins

        self.num_nodes = self.policies[0].horizon.num_nodes
        self.last_failed = []
        self.last_pos_err = np.zeros(0)
        self.last_horizon = torch.zeros(self.num_envs, self.num_drones, self.num_nodes, 12, device=base.device)
        """Every drone's setpoints over the whole horizon from the last solve,
        (num_envs, num_drones, num_nodes, 12) as p, v, a, w per node, env
        frame. Drone-major, so drone d's slice is one uniform-policy label."""
        self.last_load_horizon = np.zeros((self.num_envs, self.num_nodes, 3))
        """Where the last solve expects each env's payload over the horizon,
        (num_envs, num_nodes, 3), env frame. Diagnostic: node 1 against the
        measured payload one step later tests the MPC's model, the last node
        against the goal tests what the MPC is aiming for."""

        print(
            f"[INFO]: MPC N={self.policies[0].backend.N} nx={self.policies[0].backend.nx} "
            f"drones={self.policies[0].num_drones}"
        )
        print(f"[INFO]: solving every env step, {1.0 / self.step_dt:.0f} Hz")

        # One solve per env side by side: acados is called through ctypes,
        # which releases the GIL for the C solve, so the solves of different
        # envs overlap (each env has its own solver memory, and the results
        # are bitwise the ones a sequential loop gives). What stays serial is
        # the Python around each solve, which caps the gain at ~3.5x. How to
        # parallelise is really the backend's business: once a batched backend
        # (CusADi) exists, this pool belongs behind a batch interface in
        # python_mpc_cusadi, next to the acados backend it is specific to.
        self.solve_threads = solve_threads or min(self.num_envs, os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(self.solve_threads)
        print(f"[INFO]: solving on {self.solve_threads} threads")

    def measured_load_state(self, i: int, time: float) -> LoadState:
        """Env-frame state of env i's payload.

        Isaac Lab 3.0 quaternions are XYZW like the MPC's (they were WXYZ
        before 3.0, and permuting them again reads a yawed payload as flipped
        upside down). The angular velocity goes to the payload body frame
        here, so everything past this point is pure MPC convention.
        """
        row = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        return self._load_state(row, self.env_origins[i].cpu().numpy(), time)

    @staticmethod
    def _load_state(row, origin, time) -> LoadState:
        return LoadState(
            time=time,
            p=row[:3] - origin,
            q=row[3:7],
            v=row[7:10],
            w=Rotation.from_quat(row[3:7]).as_matrix().T @ row[10:13],
        )

    def measured_drone_states(self, i: int, time: float) -> list[DroneState]:
        """Env i's drone states, env frame like `measured_load_state`.

        `solve` only needs the positions -- the taut cables hang from the
        measured endpoints -- but the velocities come for free out of the
        same sim buffer, and `a` is a command, not a measurement.
        """
        rows = self.robot.data.body_com_state_w.torch[i, self.falcon_idx].cpu().numpy().astype(float)
        return self._drone_states(rows, self.env_origins[i].cpu().numpy(), time)

    @staticmethod
    def _drone_states(rows, origin, time) -> list[DroneState]:
        return [DroneState(time=time, p=row[:3] - origin, v=row[7:10], w=row[10:13]) for row in rows]

    def seed(self, env_ids):
        """Point each listed env's policy at its command term's current ramp.

        The command term rebuilt those ramps when the envs were reset -- the
        policy only needs to be emptied and handed the trajectory. Call after
        the reset, never before: the old reference is what the old episode was
        flying, and a policy without one cannot solve.
        """
        term = self.env.command_manager.get_term(self.COMMAND_NAME)
        for i in env_ids:
            policy = self.policies[i]
            policy.reset()  # keeps nothing; the fresh reference follows
            policy.set_reference(term.get_reference(i))

    def act(self, stime: float) -> torch.Tensor:
        """Solve all envs and return the waypoint action for the next step.

        The full horizon behind that action is left in `last_horizon`. Envs
        whose solve failed are listed in `last_failed`; their setpoints are
        still emitted (the solver's last iterate) but are not trustworthy.
        """
        # everything crossing into the policy is measured, env frame
        # (quaternions XYZW here and in the MPC, angular velocity in the
        # payload body frame); one device-to-host copy for all envs
        rows = self.robot.data.body_com_state_w.torch[:, [self.load_idx, *self.falcon_idx]].cpu().numpy()
        rows = rows.astype(float)
        origins = self.env_origins.cpu().numpy()
        measured = [
            (self._load_state(rows[i, 0], origins[i], stime), self._drone_states(rows[i, 1:], origins[i], stime))
            for i in range(self.num_envs)
        ]

        solves = list(self._pool.map(lambda i: self.policies[i].solve(stime, *measured[i]), range(self.num_envs)))

        waypoint = np.zeros(tuple(self.env.action_manager.action.shape))
        horizon = np.zeros(tuple(self.last_horizon.shape))
        pos_errs = []
        self.last_failed = []
        for i, (predicted_drones, predicted_load, status) in enumerate(solves):
            measured_load, policy = measured[i][0], self.policies[i]
            self.last_load_horizon[i] = predicted_load.p
            if status != OCPStatus.SUCCESS:
                # a failed QP means the setpoints below are not trustworthy.
                # The geometry goes out with it: which status it is says
                # little on its own, but paired with where the payload was
                # and how far it still had to go it usually says plenty.
                if self.verbose:
                    goal = policy.traj.p[-1]
                    print(
                        f"[WARN]: env {i} status {status.name} at t={stime:.2f}s, "
                        f"ramp {'running' if stime < policy.traj.time[-1] else 'done'}, "
                        f"{np.linalg.norm(measured_load.p - goal):.2f} m to goal, "
                        f"p={np.round(measured_load.p, 2)}, |v|={np.linalg.norm(measured_load.v):.2f} m/s"
                    )
                self.last_failed.append(i)

            # the whole plan, (nodes, drones, 12) from the solver, stored
            # drone-major; the setpoints to apply now are its node 1, 10 ms
            # ahead, i.e. the next environment step, with zeros after p, v, a
            plan = np.concatenate(
                [predicted_drones.p, predicted_drones.v, predicted_drones.a, predicted_drones.w], axis=-1
            ).transpose(1, 0, 2)
            horizon[i] = plan
            waypoint[i] = np.concatenate([plan[:, 1, :9], np.zeros((self.num_drones, 3))], axis=-1).reshape(-1)

            # p[-1], not p[0]: the reference is a ramp now, so its last
            # sample is the goal and its first is where the ramp began.
            pos_errs.append(float(np.linalg.norm(measured_load.p - policy.traj.p[-1])))

        self.last_pos_err = np.asarray(pos_errs)
        self.last_horizon = torch.as_tensor(horizon, dtype=self.last_horizon.dtype, device=self.env.device)
        return torch.as_tensor(waypoint, dtype=self.env.action_manager.action.dtype, device=self.env.device)
