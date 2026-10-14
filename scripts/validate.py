# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly a trained student, or the NMPC teacher with --teacher, on every test reference in one go.

Each task in TASKS is built, flown for one episode per env and closed in turn, in one simulator, the way
Isaac Lab's own environment tests run many tasks. Reports play.py's metrics per task, then all results together.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \\
        IL_cooperative_aerial_manipulation_ext/scripts/validate.py \\
        --headless --checkpoint logs/dagger/<run_name>/round_011.pt'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate a student or the teacher on every test reference.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from scripts/train.py or dagger.py.")
parser.add_argument("--teacher", action="store_true", default=False, help="Fly the NMPC teacher, as the baseline.")
parser.add_argument("--num_envs", type=int, default=1, help="Flights per task, each from its own random start.")
parser.add_argument("--seed", type=int, default=0, help="Seed for every task; the same seed gives the same starts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.teacher and args_cli.checkpoint is None:
    parser.error("--checkpoint is required unless --teacher")

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from IL_mav_carry_ext.imitation import load_checkpoint
from IL_mav_carry_ext.reporting import TextReporter, episode_numbers
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab_tasks.utils import parse_env_cfg

TASKS = [
    "Flycrane-FigureEight-v0",
    "Flycrane-FigureEightA2-v0",
    "Flycrane-FigureEightA4-v0",
    "Flycrane-Circle-v0",
    "Flycrane-Square-v0",
]


def main():
    report = TextReporter("validate")
    student = None if args_cli.teacher else load_checkpoint(args_cli.checkpoint, args_cli.device)[0]
    results = {}
    for task in TASKS:
        if not simulation_app.is_running():
            break
        sim_utils.create_new_stage()
        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
        env_cfg.seed = args_cli.seed
        # a failed teacher solve says nothing about the student's flight
        env_cfg.terminations.mpc_failed = None
        env_cfg.commands.pose_command.solve_teacher = args_cli.teacher
        env = ManagerBasedRLEnv(cfg=env_cfg)

        n = env.num_envs
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        completed = torch.zeros_like(done)
        completed_numbers = {}  # every completed flight ends on the same step, so its reset's numbers are their means

        obs, _ = env.reset()
        while simulation_app.is_running() and not done.all():
            with torch.inference_mode():
                action = obs["label"] if student is None else student.predict(obs["policy"])
                obs, _, terminated, truncated, extras = env.step(action.flatten(1))

            ended = (terminated | truncated) & ~done
            if ended.any():
                numbers = episode_numbers(extras["log"])
                ended_completed = int((ended & truncated).sum())
                report.metrics(
                    f"{task} episodes ended", {"ended": int(ended.sum()), "completed": ended_completed, **numbers}
                )
                if ended_completed:
                    completed_numbers = numbers
            completed |= ended & truncated
            done |= ended
        env.close()
        SimulationContext.clear_instance()

        results[task] = {"completed": int(completed.sum()), "envs": n, **completed_numbers}
        report.metrics(f"result {task}", results[task])

    for task, result in results.items():
        report.metrics(f"result {task}", result)


if __name__ == "__main__":
    main()
    simulation_app.close()
