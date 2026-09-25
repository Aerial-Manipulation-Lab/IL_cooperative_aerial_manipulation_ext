# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly one flycrane along a figure eight with its NMPC, optionally recording video.

Usage:

    python flycrane_env_figure_eight.py --video --rebuild
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly one flycrane along a figure eight.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=600, help="Video length, in steps.")
parser.add_argument("--rebuild", action="store_true", default=False, help="Regenerate the acados solver first.")
parser.add_argument("--amplitude", type=float, default=1.5, help="Half-width of the figure eight, m.")
parser.add_argument("--period", type=float, default=20.0, help="Time for one full figure, s.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv
from python_mpc_cusadi import DroneCfg, LoadState, PlantCfg, TeacherPolicy, TuningCfg
from python_mpc_cusadi.backends.acados_cpu import AcadosBackend
from python_mpc_cusadi.signals.specs import FigureEight

# current flycrane configuration
FLYCRANE = PlantCfg(
    load_mass=1.45,
    load_inertia=np.array([0.04, 0.05, 0.08]),
    drones=(
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, 0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, -0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([-0.28, 0.0, 0.06])),
    ),
)


def payload_state(robot, load_idx, env_origin, stime):
    """The payload as the MPC wants it: env frame, XYZW quaternion, body-frame rates."""
    load = robot.data.body_com_state_w.torch[0, load_idx].cpu().numpy().astype(float)
    quat = load[3:7]
    return LoadState(
        time=stime,
        p=load[:3] - env_origin,
        q=quat,
        v=load[7:10],
        w=Rotation.from_quat(quat).as_matrix().T @ load[10:13],
    )


def main():
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = 1
    # the MPC gives position/velocity/acceleration per drone, i.e. geometric mode
    env_cfg.actions.low_level_action.control_mode = "geometric"
    # camera: track the flycrane instead of staring at the world origin
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (8, 8, 6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="./videos",
            name_prefix=f"flycrane_figure_eight_{run_id}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO]: recording {args_cli.video_length} steps to ./videos")

    tuning = TuningCfg()
    print("-" * 80)
    print("[INFO]: compiling the MPC problem...")
    backend = AcadosBackend.for_plant(FLYCRANE, tuning, rebuild=args_cli.rebuild)
    policy = TeacherPolicy(FLYCRANE, tuning, backend)

    robot = env.unwrapped.scene["robot"]
    load_idx = robot.find_bodies("load_odometry_sensor_link")[0][0]
    step_dt = env.unwrapped.step_dt
    env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    steps = args_cli.video_length

    env.reset()

    # Centre the figure eight on wherever the payload actually is, so the run
    # starts with zero tracking error. Then the spec's p(0) is exactly its centre.
    start = payload_state(robot, load_idx, env_origin, 0.0)
    spec = FigureEight(
        center=tuple(start.p),
        amplitude=args_cli.amplitude,
        period=args_cli.period,
        duration=steps * step_dt + 1.0,
        dt=step_dt,
    )
    policy.set_reference(spec.build())

    print(f"[INFO]: MPC N={backend.N} nx={backend.nx} drones={policy.num_drones}")
    print(f"[INFO]: figure eight, amplitude {spec.amplitude} m, period {spec.period} s, "
          f"centred {np.round(spec.center, 2)}")
    print(f"[INFO]: {steps} steps at {1.0 / step_dt:.0f} Hz")
    print("-" * 80)

    stime = 0.0
    for count in range(steps):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            state = payload_state(robot, load_idx, env_origin, stime)
            # No cable_state: the cables are not measured here, so the policy
            # falls back to its own prediction.
            drones, _ = policy.solve(stime, state)
            if policy.ocp_status != 0:
                # a failed QP means the setpoints below are not trustworthy
                print(f"[WARN]: solver status {policy.ocp_status} at t={stime:.2f}s")

            # node 1 of the horizon is one environment step ahead
            waypoint = torch.zeros_like(env.unwrapped.action_manager.action)
            for d in range(policy.num_drones):
                setpoint = np.concatenate([drones.p[1, d], drones.v[1, d], drones.a[1, d], np.zeros(3)])
                waypoint[0, d * 12 : (d + 1) * 12] = torch.as_tensor(setpoint, device=env.unwrapped.device)

            env.step(waypoint)
            stime += step_dt

            if count % 50 == 0:
                ref = policy.traj.state_at(min(count, len(policy.traj) - 1))
                print(f"t={stime:6.2f}s  pos_err={np.linalg.norm(state.p - ref.p):5.2f} m  "
                      f"ref=[{ref.p[0]:5.2f} {ref.p[1]:5.2f} {ref.p[2]:5.2f}]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
