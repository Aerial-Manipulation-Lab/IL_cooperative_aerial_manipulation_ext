# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher that computes per-env drone setpoints from Isaac Lab state."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from python_mpc_cusadi import CableState, DroneCfg, LoadState, PlantCfg, TeacherPolicy, TuningCfg
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend
from python_mpc_cusadi.ocp.problem import OcpProblem
from python_mpc_cusadi.signals.specs import Approach

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

# Long enough that the goal is never more than a gentle move away: the command
# ranges reach ~2.8 m from the origin, so this is well under 1 m/s.
RAMP_SECONDS = 3.0

# acados' return codes. 2 and 4 are very different problems: 2 is a solve that
# ran out of iterations and may still be usable, 4 is one the QP could not
# solve at all.
ACADOS_STATUS = {
    0: "success",
    1: "NaN detected",
    2: "max iterations",
    3: "minimum step size",
    4: "QP solver failed",
}


class MpcTeacher:
    """One NMPC per env, solving against the scene's robot every env step.

    Owns the acados backends, the ramped references and the bookkeeping of
    which envs' last solve failed. The env is only read from, never modified:
    resetting done envs stays the wrapper's (or the caller's) job.
    """

    def __init__(self, env, plant=FLYCRANE, tuning=None, rebuild=False, ramp_seconds=RAMP_SECONDS, verbose=True):
        base = env.unwrapped
        self.env = base
        self.plant = plant
        self.ramp_seconds = ramp_seconds
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

    def load_pose(self, i: int):
        """Env-frame position and yaw of env i's payload, as `Approach` wants them."""
        row = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        # Quaternions are XYZW here, the same convention the solve loop reads.
        yaw = float(Rotation.from_quat(row[3:7]).as_euler("zyx")[0])
        return row[:3] - self.env_origins[i].cpu().numpy(), yaw

    def measured_cables(self, i: int):
        """Env i's cable directions, read off the sim rather than guessed.

        `OcpProblem.encode` treats these as measured and writes them into x0,
        which is an equality constraint. Passing nothing instead makes the
        policy fall back to its own equilibrium -- cables vertical -- and the
        asset hangs its ropes at ~0.5 rad, so every drone setpoint would carry
        a standing offset.
        """
        load = self.robot.data.body_com_state_w.torch[i, self.load_idx].cpu().numpy().astype(float)
        drones = self.robot.data.body_com_state_w.torch[i, self.falcon_idx, :3].cpu().numpy().astype(float)
        R = Rotation.from_quat(load[3:7]).as_matrix()

        states = []
        for d, drone in enumerate(self.plant.drones):
            s = load[:3] + R @ drone.attach_point - drones[d]
            states.append(CableState(s=(s / np.linalg.norm(s)).reshape(3, 1), l=drone.cable_length))
        return states

    def seed(self, env_ids, stime: float):
        """Ramp each listed env's policy from its payload's current pose to
        that env's freshly sampled goal. Returns the goals, keyed by env id.
        """
        cmd = self.env.command_manager.get_command("pose_command")
        goals = {}
        for i in env_ids:
            policy = self.policies[i]
            policy.reset()  # keeps nothing; a fresh goal follows
            # after a reset the payload is back at its spawn pose, which is
            # where the new ramp has to start
            start_pos, start_yaw = self.load_pose(i)
            goal = cmd[i][:3].cpu().numpy().astype(float)
            policy.set_reference(
                Approach(
                    start=tuple(start_pos),
                    goal=tuple(goal),
                    start_yaw=start_yaw,
                    goal_yaw=start_yaw,
                    duration=self.ramp_seconds,
                    time_offset=stime,
                ).build()
            )
            goals[i] = goal
        return goals

    def act(self, stime: float) -> torch.Tensor:
        """Solve all envs and return the waypoint action for the next step.

        Envs whose solve failed are listed in `last_failed`; their setpoints
        are still emitted (the solver's last iterate) but are not trustworthy.
        """
        waypoint = torch.zeros_like(self.env.action_manager.action)
        # payload state, env frame. Quaternions are XYZW here and in the MPC,
        # but the angular velocity has to go to the payload body frame.
        load_state_w = self.robot.data.body_com_state_w.torch[:, self.load_idx]

        pos_errs = []
        self.last_failed = []
        for i in range(self.num_envs):
            load = load_state_w[i].cpu().numpy().astype(float)
            quat = load[3:7]
            state = LoadState(
                time=stime,
                p=load[:3] - self.env_origins[i].cpu().numpy(),
                q=quat,
                v=load[7:10],
                w=Rotation.from_quat(quat).as_matrix().T @ load[10:13],
            )

            policy = self.policies[i]
            # The cables are observable in sim, so measure them rather than
            # letting the policy assume they hang vertically.
            drones, _ = policy.solve(stime, state, self.measured_cables(i))
            if policy.ocp_status != 0:
                # a failed QP means the setpoints below are not trustworthy.
                # The geometry goes out with it: which status it is says
                # little on its own, but paired with where the payload was
                # and how far it still had to go it usually says plenty.
                if self.verbose:
                    goal = policy.traj.p[-1]
                    print(
                        f"[WARN]: env {i} status {policy.ocp_status} "
                        f"({ACADOS_STATUS.get(policy.ocp_status, 'unknown')}) at t={stime:.2f}s, "
                        f"ramp {'running' if stime < policy.traj.time[-1] else 'done'}, "
                        f"{np.linalg.norm(state.p - goal):.2f} m to goal, "
                        f"p={np.round(state.p, 2)}, |v|={np.linalg.norm(state.v):.2f} m/s"
                    )
                self.last_failed.append(i)

            # node 1 of the horizon is 10 ms ahead, i.e. the next environment step
            for d in range(self.num_drones):
                setpoint = np.concatenate([drones.p[1, d], drones.v[1, d], drones.a[1, d], np.zeros(3)])
                waypoint[i, d * 12 : (d + 1) * 12] = torch.as_tensor(setpoint, device=self.env.device)

            # p[-1], not p[0]: the reference is a ramp now, so its last
            # sample is the goal and its first is where the ramp began.
            pos_errs.append(float(np.linalg.norm(state.p - policy.traj.p[-1])))

        self.last_pos_err = np.asarray(pos_errs)
        return waypoint
