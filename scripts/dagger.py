# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger with the NMPC teacher.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-FigureEight-v0 (random); to test: Flycrane-FigureEightA2-v0, -FigureEightA4-v0, -Circle-v0, -Square-v0.",
)
parser.add_argument("--run_name", type=str, required=True, help="Run dir under logs/dagger; must not exist yet.")
parser.add_argument("--iterations", type=int, default=25, help="Rounds of ROUND_FLIGHT_S flight, then training.")
parser.add_argument("--num_envs", type=int, default=8, help="Flycranes flying in parallel.")
parser.add_argument("--epochs", type=int, default=16, help="Passes over every example so far, per round.")
parser.add_argument("--keep_every", type=int, default=10, help="Train on every Nth step; the recording keeps all.")
parser.add_argument("--model", choices=["mlp"], default="mlp", help="Student network, see build_model.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import time

import torch
from IL_mav_carry_ext.imitation import Dagger, build_model, save_checkpoint
from IL_mav_carry_ext.mpc import MpcRecorderManagerCfg, dataset_metadata
from IL_mav_carry_ext.reporting import EpisodeLog, TensorBoardReporter, TextReporter
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

BATCH_SIZE = 256
LR = 1e-3
GATE = 0.03  # the thesis's 0.02 MSE over 20 state columns per node, rescaled to our 12
ROUND_FLIGHT_S = 40.0
LOG_DIR = "logs/dagger"


def main():
    report = TextReporter("dagger")
    run_dir = os.path.abspath(os.path.join(LOG_DIR, args_cli.run_name))
    os.makedirs(run_dir)  # every run starts fresh, so its name must be new

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 0
    env_cfg.recorders = MpcRecorderManagerCfg(dataset_export_dir_path=run_dir, dataset_filename="rollouts")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    teacher = env.command_manager.get_term("pose_command").teacher
    meta = dataset_metadata(env)
    with open(os.path.join(run_dir, "rollouts.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    model = build_model(meta, args_cli.model).to(env.device)
    dagger = Dagger(model, LR, GATE, args_cli.keep_every, args_cli.epochs, BATCH_SIZE)
    num_envs = env.num_envs
    steps = round(ROUND_FLIGHT_S / (num_envs * env.step_dt))

    tensorboard = TensorBoardReporter(run_dir)
    report.info(f"{args_cli.iterations} rounds of {steps} steps on {num_envs} envs; logging to {run_dir}")

    obs, _ = env.reset()
    # cut each env's first episode short by a different 1/num_envs, so from then on they fly
    # different parts of it and every round holds the whole episode once
    env.episode_length_buf[:] = torch.arange(num_envs, device=env.device) * int(env.max_episode_length) // num_envs
    episodes = EpisodeLog()
    for rnd in range(args_cli.iterations):
        if not simulation_app.is_running():
            break
        round_start = time.time()
        for _ in range(steps):
            obs, _, terminated, truncated, extras = env.step(dagger.act(obs, teacher.ok))
            episodes.add(extras, terminated | truncated)
        metrics = {**dagger.update(), "round_min": (time.time() - round_start) / 60, **episodes.means()}
        report.metrics(f"round {rnd}/{args_cli.iterations}", metrics)
        tensorboard.metrics(metrics, rnd)
        save_checkpoint(os.path.join(run_dir, f"round_{rnd:03d}.pt"), model, meta, round=rnd, metrics=metrics)

    tensorboard.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
