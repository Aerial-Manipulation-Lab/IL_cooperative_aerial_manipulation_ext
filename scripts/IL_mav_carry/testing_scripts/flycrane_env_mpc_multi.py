# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly N flycranes to N independent random payload poses, each with its own NMPC."""

"""Launch Isaac Sim Simulator first."""

import argparse

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Fly N flycranes with N independent NMPCs.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
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

from datetime import datetime

import gymnasium as gym
from IL_mav_carry_ext.tasks.managerbased.flycrane.flycrane_env_cfg import ApproachPoseEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """Main function."""
    # create environment config
    env_cfg = ApproachPoseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # the MPC gives position/velocity/acceleration per drone, i.e. geometric mode
    env_cfg.actions.low_level_action.control_mode = "geometric"
    # camera: track the flycrane instead of staring at the world origin from 7.5m out.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.commands.pose_command.rebuild = args_cli.rebuild

    base = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    command = base.command_manager.get_term("pose_command")
    env = base
    if args_cli.video:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_kwargs = {
            "video_folder": "./videos",
            "name_prefix": f"flycrane_env_mpc_multi_{run_id}",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during execution.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(base, **video_kwargs)

    print("-" * 80)
    env.reset()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(command.teacher.action)
        step += 1

        if step % 50 == 0:
            pos_err = command.metrics["position_error"]
            print(
                f"t={step * base.step_dt:6.2f}s  pos_err mean={pos_err.mean():5.2f} m  "
                f"max={pos_err.max():5.2f} m  (across {base.num_envs} envs)"
            )

        if args_cli.video and step == args_cli.video_length:
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
