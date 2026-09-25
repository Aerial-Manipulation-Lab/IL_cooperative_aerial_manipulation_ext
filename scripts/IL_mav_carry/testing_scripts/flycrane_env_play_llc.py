# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--control_mode", type=str, default="geometric", help="Control mode for the agent.")

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
import math

from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    # the action term's mode determines action_dim (geometric: 12/drone, ACCBR: 5/drone),
    # so it has to match the waypoints written below
    env_cfg.actions.low_level_action.control_mode = args_cli.control_mode
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

    robot_mass = env.unwrapped.scene["robot"].root_physx_view.get_masses().numpy().sum()

    gravity = torch.tensor(env.unwrapped.sim.cfg.gravity, device=env.unwrapped.sim.device).norm()
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14
    payload_mass = 1.4 + 0.00001 + 0.006
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass

    stretch_position = torch.tensor(
        [
            [
                1.27,
                1.0867,
                1.7,  # drone 1
                1.27,
                -1.0867,
                1.7,  # drone 2
                -0.1367,
                0.0,
                1.7,
            ]
        ],
        dtype=torch.float32,
    )

    straight_up_position = torch.tensor(
        [
            [
                0.27,
                0.22,
                2.141,  # drone 1
                0.27,
                -0.22,
                2.141,  # drone 2
                -0.27,
                0.0,
                2.141,
            ]
        ],
        dtype=torch.float32,
    )

    ACC_BR_ref = torch.tensor(
        [
            [
                0.0,
                0.0,
                9.0,
                0.0,
                0.0,
                10.0,  # drone 1
                0.0,
                0.0,
                8.0,
                0.0,
                0.0,
                10.0,  # drone 2
                0.0,
                0.0,
                7.0,
                0.0,
                0.0,
                10.0,
            ]
        ],
        dtype=torch.float32,
    )

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            falcon_pos = env.unwrapped.scene["robot"].data.body_com_state_w.torch[:, [20, 27, 34], :3]
            # reset
            if count % 500 == 0:
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            waypoint = torch.zeros_like(env.unwrapped.action_manager.action)
            # When using geometric
            if args_cli.control_mode == "geometric":
                waypoint[:, :3] = stretch_position[:, :3]
                waypoint[:, 12:15] = stretch_position[:, 3:6]
                waypoint[:, 24:27] = stretch_position[:, 6:9]

            # waypoint[:, :3] = falcon_pos[:, 0]
            # waypoint[:, 12:15] = falcon_pos[:, 1]
            # waypoint[:, 24:27] = falcon_pos[:, 2]

            # when using ACCBR
            if args_cli.control_mode == "ACCBR":
                waypoint[:] = ACC_BR_ref
            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            count += 1

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
