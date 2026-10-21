# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import os
import shutil

from isaaclab.app import AppLauncher

LOG_DIR = "logs/dagger"

parser = argparse.ArgumentParser(description="DAgger with the NMPC teacher.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-RandomTrajectory-v0, -RandomTrajectoryMediumToFast-v0, Flycrane-FigureEight-v0 (random), Flycrane-FigureEightSlow-v0, -FigureEightMedium-v0, -FigureEightMediumPlus-v0, -FigureEightFast-v0, Flycrane-Circle-v0, Flycrane-Square-v0.",
)
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Name of directory under logs/dagger. Stores run in this directory.",
)
parser.add_argument(
    "--overwrite", action="store_true", default=False, help="Replace an existing run of that name."
)
parser.add_argument(
    "--iterations",
    type=int,
    default=25,
    help="Number dagger rounds. Each round trains on a full flight, staggered by the number of envs.",
)
parser.add_argument(
    "--num_envs", type=int, default=8, help="Number of flycranes flying in parallel."
)
parser.add_argument(
    "--epochs", type=int, default=16, help="Passes over every example aggregated so far, per round."
)
parser.add_argument(
    "--keep_every",
    type=int,
    default=10,
    help="Train on every Nth simulation step. Each step is recorded into a recorded dataset regardles.",
)
parser.add_argument(
    "--model", choices=["mlp"], default="mlp", help="Student network, see build_model."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

run_dir = os.path.abspath(os.path.join(LOG_DIR, args_cli.run_name))
if os.path.exists(run_dir):
    if not args_cli.overwrite:
        raise SystemExit(f"[ERROR]: {run_dir} exists. Pass --overwrite to replace it.")
    shutil.rmtree(run_dir)
os.makedirs(run_dir)

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import time

import torch
from IL_mav_carry_ext.imitation import Dagger, DaggerCfg, FeatureSpec, build_model, save_checkpoint
from IL_mav_carry_ext.mpc import MpcRecorderManagerCfg
from IL_mav_carry_ext.reporting import EpisodeLog, TensorBoardReporter, TextReporter
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

ROUND_FLIGHT_S = 40.0
ROUND_LINE = (
    "student_share",
    "gate",
    "train_loss",
    "val_loss",
    "node1_vel_err_mps",
    "node1_acc_err_mps2",
)
"""The round's numbers the console shows, in order; TensorBoard gets them all."""


def main():
    # create textual logger
    report = TextReporter("dagger")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env_cfg.recorders = MpcRecorderManagerCfg(
        dataset_export_dir_path=run_dir, dataset_filename="rollouts"
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)
    teacher = env.command_manager.get_term("pose_command").teacher
    spec = FeatureSpec.from_env(env)
    with open(os.path.join(run_dir, "rollouts.meta.json"), "w") as f:
        json.dump(spec.to_dict(), f, indent=2)

    model = build_model(spec, args_cli.model).to(env.device)
    dagger = Dagger(DaggerCfg(keep_every=args_cli.keep_every, epochs=args_cli.epochs), model)
    num_envs = env.num_envs
    steps = round(ROUND_FLIGHT_S / (num_envs * env.step_dt))

    tensorboard = TensorBoardReporter(run_dir)
    report.info(
        f"{args_cli.iterations} rounds of {steps} steps on {num_envs} envs; logging to {run_dir}"
    )

    obs, _ = env.reset()
    # cut each env's first episode short by a different 1/num_envs, so from then on they fly
    # different parts of it and every round holds the whole episode once
    env.episode_length_buf[:] = (
        torch.arange(num_envs, device=env.device) * int(env.max_episode_length) // num_envs
    )
    episodes = EpisodeLog()
    for rnd in range(args_cli.iterations):
        if not simulation_app.is_running():
            break
        round_start = time.time()
        for _ in range(steps):
            obs, _, terminated, truncated, extras = env.step(dagger.act(obs, teacher.ok))
            episodes.add(extras, terminated | truncated)
        metrics = {
            **dagger.update(),
            "round_min": (time.time() - round_start) / 60,
            **episodes.means(),
        }
        report.metrics(
            f"round {rnd}/{args_cli.iterations}",
            {k: metrics[k] for k in ROUND_LINE if k in metrics},
        )
        tensorboard.metrics(metrics, rnd)
        save_checkpoint(
            os.path.join(run_dir, f"round_{rnd:03d}.pt"), model, spec, round=rnd, metrics=metrics
        )

    tensorboard.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
