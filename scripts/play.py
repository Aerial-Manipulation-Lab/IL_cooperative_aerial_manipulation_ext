# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly one trained student for a single episode.

The error is measured against the NMPC teacher.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/play.py \
        --headless --checkpoint logs/bc/<run_name>/best.pt --video'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly one trained student for a single episode.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint from scripts/train.py.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the task env; the same seed samples the same goals.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the flight.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch
from IL_mav_carry_ext.imitation import StudentPolicy
from IL_mav_carry_ext.mpc import MpcTeacherWrapper
from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc
from isaaclab.envs import ManagerBasedRLEnv

PRINT_S = 0.5


def print_heartbeat(t, last_err, err_sum, err_steps):
    print(
        f"[t={t:5.1f} s] label err {last_err * 100:6.2f} cm (mean {err_sum / max(err_steps, 1) * 100:.2f} cm)",
        flush=True,
    )


def main():
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    base = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    teacher_env = MpcTeacherWrapper(base, beta=0.0, reset_on_failed_solve=False, verbose=False)
    env = teacher_env
    if args_cli.video:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        env = gym.wrappers.RecordVideo(
            teacher_env,
            video_folder="./videos",
            name_prefix=f"play_{name}_{run_id}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    student = StudentPolicy(args_cli.checkpoint, base.device)
    num_drones = teacher_env.teacher.num_drones

    err_sum = err_steps = 0.0
    last_err = float("nan")
    failed_solves = 0

    obs, _ = env.reset()
    steps = 0
    next_print = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            action = student(obs["policy"])
            obs, _, terminated, truncated, info = env.step(action)

        steps += 1
        failed_solves += len(info["mpc_failed"])

        if bool(base.mpc_ok[0]):
            sent = action.view(1, num_drones, 12)[..., :3]
            label = base.teacher_action.view(1, num_drones, 12)[..., :3]
            last_err = float((sent - label).norm(dim=-1).mean())
            err_sum += last_err
            err_steps += 1

        t = steps * base.step_dt
        if t >= next_print:
            next_print += PRINT_S
            print_heartbeat(t, last_err, err_sum, err_steps)

        if bool((terminated | truncated)[0]):
            break
        if args_cli.video and teacher_env.count >= args_cli.video_length:
            break

    print(
        f"\n[RESULT] student, {steps * base.step_dt:.1f} s flown | "
        f"label err mean {err_sum / max(err_steps, 1) * 100:.2f} cm | "
        f"failed solves {failed_solves}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
