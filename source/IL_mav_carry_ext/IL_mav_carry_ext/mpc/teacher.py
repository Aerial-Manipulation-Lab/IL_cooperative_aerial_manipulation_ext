# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher that computes per-env drone setpoints from Isaac Lab state."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from python_mpc_cusadi import (DroneCfg, DroneState, LoadState, OCPStatus,
                               PlantCfg, TeacherPolicy, TuningCfg)
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend
from python_mpc_cusadi.ocp.problem import OcpProblem

# the flycrane as measured on the hardware, same as examples/run_figure_eight.py
FLYCRANE = PlantCfg(
    load_mass=1.45,
    load_inertia=np.array([0.04, 0.05, 0.08]),
    drones=(
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, 0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, -0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([-0.28, 0.0, 0.06])),
    ),
)


class MpcTeacher:
    """One NMPC per env, solving against the scene's robot every env step.

    Owns the acados backends and the bookkeeping of which envs' last solve
    failed; references come from the task's command term, which owns their
    lifecycle. The env is only read from, never modified: resetting done envs
    stays the wrapper's (or the caller's) job.
    """

    COMMAND_NAME = "pose_command"
    """Command term the ramped references are read from."""

    def __init__(self, env, plant=FLYCRANE, tuning=None, rebuild=False, verbose=True):
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

        self.last_failed = []
        self.last_pos_err = np.zeros(0)

        print(f"[INFO]: MPC N={self.policies[0].backend.N} nx={self.policies[0].backend.nx} "
              f"drones={self.policies[0].num_drones}")
        print(f"[INFO]: solving every env step, {1.0 / self.step_dt:.0f} Hz")

    def measured_load_state(self, i: int, time: float) -> LoadState:
        """Env-frame state of env i's payload.

        Isaac Lab 3.0 quaternions are XYZW like the MPC's (they were WXYZ
        before 3.0, and permuting them again reads a yawed payload as flipped
        upside down). The angular velocity goes to the payload body frame
        here, so everything past this point is pure MPC convention.
        """
        row = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        return LoadState(
            time=time,
            p=row[:3] - self.env_origins[i].cpu().numpy(),
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
        origin = self.env_origins[i].cpu().numpy()
        return [DroneState(time=time, p=row[:3] - origin, v=row[7:10], w=row[10:13])
                for row in rows]

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

        Envs whose solve failed are listed in `last_failed`; their setpoints
        are still emitted (the solver's last iterate) but are not trustworthy.
        """
        waypoint = torch.zeros_like(self.env.action_manager.action)

        pos_errs = []
        self.last_failed = []
        for i in range(self.num_envs):
            # everything crossing into the policy is measured, env frame
            # (quaternions XYZW here and in the MPC, angular velocity in the
            # payload body frame)
            measured_load = self.measured_load_state(i, stime)
            measured_drones = self.measured_drone_states(i, stime)

            policy = self.policies[i]
            predicted_drones, _, status = policy.solve(stime, measured_load, measured_drones)
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

            # the setpoints to apply now: horizon node 1, 10 ms ahead, i.e.
            # the next environment step
            for d, next_setpoint in enumerate(predicted_drones.state_next):
                setpoint = np.concatenate([next_setpoint.p, next_setpoint.v,
                                           next_setpoint.a, np.zeros(3)])
                waypoint[i, d * 12 : (d + 1) * 12] = torch.as_tensor(setpoint, device=self.env.device)

            # p[-1], not p[0]: the reference is a ramp now, so its last
            # sample is the goal and its first is where the ramp began.
            pos_errs.append(float(np.linalg.norm(measured_load.p - policy.traj.p[-1])))

        self.last_pos_err = np.asarray(pos_errs)
        return waypoint
