# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly the flycrane to one fixed payload pose with the NMPC from python_mpc_cusadi.

Same setup as flycrane_env_play_llc.py, but the waypoints come from the MPC
instead of being hardcoded. The MPC is the outer loop -- it plans the payload
and the cables and places the drones by flatness -- so the action term runs in
"geometric" mode and gets position/velocity/acceleration per drone (jerk stays
zero, the MPC has none).

Pass --rebuild once per machine: acados artifacts embed absolute paths.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Fly the flycrane with the NMPC.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rebuild", action="store_true", default=False, help="Regenerate the acados solver first.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from python_mpc_cusadi import CableState, DroneCfg, LoadState, PlantCfg, TeacherPolicy, Trajectory, TuningCfg
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend

# where the payload should go: position in the env frame, orientation XYZW
GOAL_POS = np.array([0.0, 0.0, 1.5])
GOAL_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

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


def hover_trajectory():
    """The goal pose as a two-sample trajectory. The sampler clamps outside it."""
    tile = lambda v: np.repeat(np.asarray(v, dtype=float).reshape(1, -1), 2, axis=0)
    return Trajectory(
        time=np.array([0.0, 1e4]),
        p=tile(GOAL_POS),
        q=tile(GOAL_QUAT),
        v=np.zeros((2, 3)),
        a=np.zeros((2, 3)),
        w=np.zeros((2, 3)),
        alpha=np.zeros((2, 3)),
    )


def cable_states(policy, stime, quat):
    """What the cables are doing: the previous solve's prediction, or equilibrium.

    Only the directions are observable; rate, acceleration and tension come from
    the MPC's own prediction, which is what the ROS wrapper did too. `l` has to
    be set here -- the decoder places the drones at `p + R*rho - l*s`, so the
    zero that CableTrajectory.at() leaves would park every setpoint on its
    attachment point.
    """
    if policy.cable_prediction is None:
        s_ref, t_ref, _ = policy.allocator.allocate(
            Rotation.from_quat(quat).as_matrix(), policy.params.external_wrench[0]
        )
        states = [CableState(s=s_ref[i].reshape(3, 1), t=float(t_ref[i])) for i in range(policy.num_drones)]
    else:
        states = policy.cable_prediction.at(stime)
    for i, cs in enumerate(states):
        cs.l = policy.plant.drones[i].cable_length
    return states


def fired_terminations(env):
    """Which termination terms ended the episode. Diagnostic only, so it never raises."""
    try:
        manager = env.unwrapped.termination_manager
        return [name for name in manager.active_terms if bool(manager.get_term(name)[0])] or ["(none)"]
    except Exception as exc:  # the manager API is not worth crashing a run over
        return [f"(unavailable: {exc})"]


def print_geometry(load_p, load_q, drone_p):
    """Configured cable lengths against what the simulation shows.

    The flatness map places the drones with these numbers, so a wrong attach
    point or cable length is a wrong setpoint -- which looks like a controller
    that flies but holds a standing offset.
    """
    R = Rotation.from_quat(load_q).as_matrix()
    print("[INFO]: rig geometry, configured vs. simulated")
    for i, drone in enumerate(FLYCRANE.drones):
        measured = float(np.linalg.norm(load_p + R @ drone.attach_point - drone_p[i]))
        print(
            f"         drone {i}: cable_length={drone.cable_length:.3f} m  "
            f"|attach - drone|={measured:.3f} m  delta={measured - drone.cable_length:+.3f} m"
        )


def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    # the MPC gives position/velocity/acceleration per drone, i.e. geometric mode
    env_cfg.actions.low_level_action.control_mode = "geometric"
    # camera: track the flycrane instead of staring at the world origin from 7.5m out.
    # eye/lookat are relative to the asset root, so shrink eye to zoom in.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (3.5, 3.5, 5.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    tuning = TuningCfg()
    policy = TeacherPolicy(FLYCRANE, tuning, AcadosBackend.for_plant(FLYCRANE, tuning, rebuild=args_cli.rebuild))
    policy.set_reference(hover_trajectory())

    robot = env.unwrapped.scene["robot"]
    load_idx = robot.find_bodies("load_odometry_sensor_link")[0][0]
    # the frame the geometric controller regulates, so the frame the setpoint means
    falcon_idx = robot.find_bodies("Falcon.*_base_link_inertia")[0]
    step_dt = env.unwrapped.step_dt

    print("-" * 80)
    print(f"[INFO]: MPC N={policy.backend.N} nx={policy.backend.nx} drones={policy.num_drones}")
    print(f"[INFO]: goal {np.round(GOAL_POS, 3)} quat {np.round(GOAL_QUAT, 3)} (env frame, XYZW)")
    print(f"[INFO]: solving every env step, {1.0 / step_dt:.0f} Hz")
    print("-" * 80)

    stime = 0.0
    count = 0
    env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # payload state, env frame. Quaternions are XYZW here and in the MPC,
            # but the angular velocity has to go to the payload body frame.
            load = robot.data.body_com_state_w.torch[0, load_idx].cpu().numpy().astype(float)
            quat = load[3:7]
            state = LoadState(
                time=stime,
                p=load[:3] - env.unwrapped.scene.env_origins[0].cpu().numpy(),
                q=quat,
                v=load[7:10],
                w=Rotation.from_quat(quat).as_matrix().T @ load[10:13],
            )

            drone_p = (
                robot.data.body_com_state_w.torch[0, falcon_idx, :3] - env.unwrapped.scene.env_origins[0]
            ).cpu().numpy().astype(float)

            if count == 0:
                print_geometry(state.p, quat, drone_p)

            drones, _ = policy.solve(stime, state, cable_states(policy, stime, quat))
            if policy.ocp_status != 0:
                # a failed QP means the setpoints below are not trustworthy
                print(f"[WARN]: solver status {policy.ocp_status} at t={stime:.2f}s")

            # node 1 of the horizon is 10 ms ahead, i.e. the next environment step
            waypoint = torch.zeros_like(env.unwrapped.action_manager.action)
            for i in range(policy.num_drones):
                setpoint = np.concatenate([drones.p[1, i], drones.v[1, i], drones.a[1, i], np.zeros(3)])
                waypoint[:, i * 12 : (i + 1) * 12] = torch.as_tensor(setpoint, device=env.unwrapped.device)

            if count % 50 == 0:
                # pos_err is the MPC's job, track_err the inner loop's: a large
                # pos_err with a small track_err means the plan is wrong, the
                # other way round means the drones cannot fly the plan.
                pos_err = float(np.linalg.norm(state.p - GOAL_POS))
                ori_err = float(np.degrees((Rotation.from_quat(quat).inv() * Rotation.from_quat(GOAL_QUAT)).magnitude()))
                track_err = float(np.max(np.linalg.norm(drones.p[1] - drone_p, axis=-1)))
                solve_ms = 1e3 * (policy.t_encode_sample + policy.t_solver + policy.t_decode)
                print(
                    f"t={stime:6.2f}s  status={policy.ocp_status}  pos_err={pos_err:5.2f} m  "
                    f"ori_err={ori_err:5.1f} deg  track_err={track_err:5.2f} m  solve={solve_ms:5.1f} ms"
                )

            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            stime += step_dt
            count += 1

            # the env auto-resets on termination; the MPC's warm start is then stale
            if bool(terminated[0]) or bool(truncated[0]):
                print(
                    f"[INFO]: episode ended at t={stime:.2f}s "
                    f"({', '.join(fired_terminations(env))}), resetting the MPC"
                )
                policy.reset()  # keeps the reference, drops the warm start

            if args_cli.video:
                if count == args_cli.video_length:
                    break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
