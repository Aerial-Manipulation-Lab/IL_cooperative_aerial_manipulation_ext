# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly a trained student, or the NMPC teacher with --teacher, for one episode per env.

Reports the thesis's metrics, which the env computes (`MpcCommand`'s metrics, Isaac Lab's termination
shares): the payload's position and attitude RMSE against its reference, and how each episode ended,
with two drones closer than 0.4 m ending it as a failure, as in the thesis.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/play.py \
        --headless --task Flycrane-FigureEightMedium-v0 --checkpoint logs/dagger/<run_name>/round_024.pt --num_envs 8'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly a student or the teacher for one episode per env.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-RandomTrajectory-v0, Flycrane-FigureEight-v0 (random); to test: Flycrane-FigureEightSlow-v0, -FigureEightMedium-v0, -FigureEightMediumPlus-v0, -FigureEightFast-v0, -Circle-v0, -Square-v0.",
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
import torch
from IL_mav_carry_ext.imitation import FeatureSpec, load_checkpoint
from IL_mav_carry_ext.reporting import Heartbeat, TextReporter, episode_numbers
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

PROGRESS_S = 10.0


def main():
    report = TextReporter("play")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # a failed teacher solve says nothing about the student's flight
    env_cfg.terminations.teacher_solve_failed = None
    env_cfg.commands.pose_command.solve_teacher = args_cli.teacher

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    base = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    command = base.command_manager.get_term("pose_command")
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
    student = None
    if not args_cli.teacher:
        student, spec, _ = load_checkpoint(args_cli.checkpoint, base.device)
        if spec != FeatureSpec.from_env(base):
            raise ValueError(f"{args_cli.checkpoint} was trained on other rows than {args_cli.task} gives")

    n = base.num_envs
    done = torch.zeros(n, dtype=torch.bool, device=base.device)
    completed = torch.zeros_like(done)
    completed_numbers = {}  # every completed flight ends on the same step, so its reset's numbers are their means

    obs, _ = env.reset()
    progress = Heartbeat(PROGRESS_S)
    step = 0
    while simulation_app.is_running() and not done.all():
        with torch.inference_mode():
            action = obs["label"] if student is None else student.predict(obs["policy"])
            obs, _, terminated, truncated, extras = env.step(action.flatten(1))
        step += 1

        ended = (terminated | truncated) & ~done
        if ended.any():
            numbers = episode_numbers(extras["log"])
            ended_completed = int((ended & truncated).sum())
            report.metrics("episodes ended", {"ended": int(ended.sum()), "completed": ended_completed, **numbers})
            if ended_completed:
                completed_numbers = numbers
        completed |= ended & truncated
        done |= ended
        if progress.due() and not done.all():
            report.metrics(
                "progress",
                {
                    "t_s": step * base.step_dt,
                    "ended": int(done.sum()),
                    "pos_rmse_m": float(command.metrics["pos_rmse_m"][~done].mean()),
                },
            )

    report.metrics(f"result {name}", {"completed": int(completed.sum()), "envs": n, **completed_numbers})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
