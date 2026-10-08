# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly a trained student, or the NMPC teacher with --teacher, for one episode per env.

Reports the thesis's metrics on the payload against its reference, sampled every 0.1 s: position
RMSE and attitude RMSE, plus the closest two drones came and whether the episode was completed.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/play.py \
        --headless --task Flycrane-FigureEightA2-v0 --checkpoint logs/dagger/<run_name>/round_024.pt --num_envs 8'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly a student or the teacher for one episode per env.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-FigureEight-v0 (random), Flycrane-FigureEightA2-v0 or Flycrane-FigureEightA4-v0.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from scripts/train.py or dagger.py.")
parser.add_argument("--teacher", action="store_true", default=False, help="Fly the NMPC teacher, as the baseline.")
parser.add_argument("--num_envs", type=int, default=1, help="Flights in parallel, each from its own random start.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the task env; the same seed samples the same goals.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the flight.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.teacher and args_cli.checkpoint is None:
    parser.error("--checkpoint is required unless --teacher")
if args_cli.video:
    args_cli.enable_cameras = True

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from IL_mav_carry_ext.imitation import StudentPolicy
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

PRINT_S = 5.0


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # a failed teacher solve says nothing about the student's flight
    env_cfg.terminations.mpc_failed = None
    env_cfg.commands.pose_command.solve_teacher = args_cli.teacher

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    base = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    command = base.command_manager.get_term("pose_command")
    teacher = command.teacher
    name = "teacher" if args_cli.teacher else os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    env = base
    if args_cli.video:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            base,
            video_folder="./videos",
            name_prefix=f"play_{name}_{run_id}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    student = None if args_cli.teacher else StudentPolicy.load(args_cli.checkpoint, base.device)

    n, dt = base.num_envs, base.step_dt
    every = round(0.1 / dt)
    pairs = np.triu_indices(teacher.num_drones, 1)
    origins = base.scene.env_origins.cpu().numpy()
    pos_sq, ang_sq, samples, flown = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    closest = np.full(n, np.inf)
    done, completed = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    obs, _ = env.reset()
    step = 0
    while simulation_app.is_running() and not done.all():
        state = teacher.robot.data.body_com_state_w.torch.cpu().numpy()
        drones = state[:, teacher.falcon_idx, :3]
        gap = np.linalg.norm(drones[:, pairs[0]] - drones[:, pairs[1]], axis=-1).min(1)
        closest = np.where(done, closest, np.minimum(closest, gap))
        if step % every == 0:
            now = base.common_step_counter * dt
            for i in np.flatnonzero(~done):
                ref = command.get_reference(i)
                k = ref.indices_at(now - dt / 2)
                load = state[i, teacher.load_idx]
                pos_sq[i] += np.sum((load[:3] - origins[i] - ref.p[k]) ** 2)
                ang_sq[i] += np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(load[3:7], ref.q[k])))))) ** 2
                samples[i] += 1

        with torch.inference_mode():
            action = teacher.action if student is None else student(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(action)
        step += 1

        ended = (terminated | truncated).cpu().numpy() & ~done
        completed |= ended & truncated.cpu().numpy()
        flown[ended] = step * dt
        done |= ended
        if step % round(PRINT_S / dt) == 0:
            live = samples > 0
            print(
                f"[t={step * dt:5.1f} s] {int(done.sum())}/{n} ended | position RMSE so far "
                f"{np.sqrt(pos_sq[live] / samples[live]).mean():.3f} m",
                flush=True,
            )

    pos = np.sqrt(pos_sq / np.maximum(samples, 1))
    ang = np.sqrt(ang_sq / np.maximum(samples, 1))
    print()
    for i in range(n):
        print(
            f"  env {i}: {'completed' if completed[i] else 'crashed'} after {flown[i]:.1f} s | "
            f"position RMSE {pos[i]:.3f} m | attitude RMSE {ang[i]:.2f} deg | closest drones {closest[i]:.2f} m"
        )
    c = completed
    print(
        f"[RESULT] {name}: {int(c.sum())}/{n} completed | position RMSE {pos[c].mean():.3f} m | "
        f"attitude RMSE {ang[c].mean():.2f} deg (means over completed flights) | closest drones {closest.min():.2f} m",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
