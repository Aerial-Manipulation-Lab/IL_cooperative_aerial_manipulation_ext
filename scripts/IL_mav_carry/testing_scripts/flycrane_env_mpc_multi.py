# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly N flycranes to N independent random payload poses, each with its own NMPC.
"""

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
import numpy as np

from IL_mav_carry_ext.mpc import MpcTeacherWrapper
from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    # the MPC gives position/velocity/acceleration per drone, i.e. geometric mode
    env_cfg.actions.low_level_action.control_mode = "geometric"
    # camera: track the flycrane instead of staring at the world origin from 7.5m out.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # the NMPC teacher is the policy; reset() also seeds each env's ramped goal
    env = MpcTeacherWrapper(env, rebuild=args_cli.rebuild)
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
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print("-" * 80)
    num_envs = env.unwrapped.num_envs
    step_dt = env.unwrapped.step_dt

    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step()

            if env.count % 50 == 0:
                pos_err = env.teacher.last_pos_err
                print(
                    f"t={env.time:6.2f}s  pos_err mean={np.mean(pos_err):5.2f} m  "
                    f"max={np.max(pos_err):5.2f} m  (across {num_envs} envs)"
                )

            if args_cli.video and env.count == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
