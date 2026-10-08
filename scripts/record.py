# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record NMPC teacher demonstrations to an HDF5 dataset.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/record.py \
        --headless --task Flycrane-ApproachPose-v0 --num_envs 8 --num_episodes 200 --output datasets/mpc_demos'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record NMPC teacher demonstrations.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-FigureEight-v0 (random), Flycrane-FigureEightA2-v0 or Flycrane-FigureEightA4-v0.",
)
parser.add_argument("--num_envs", type=int, default=8, help="Flycranes flying in parallel.")
parser.add_argument("--num_episodes", type=int, default=200, help="Stop once this many episodes are written.")
parser.add_argument(
    "--output",
    type=str,
    default="datasets/mpc_demos",
    help="Dataset path without extension; .hdf5 and .meta.json are added.",
)
parser.add_argument("--seed", type=int, default=0, help="Env seed; use a different one per run to merge.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Replace an existing dataset.")
parser.add_argument("--rebuild", action="store_true", default=False, help="Regenerate the acados solver first.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import os

output = os.path.abspath(os.path.splitext(args_cli.output)[0])
if os.path.exists(output + ".hdf5") and not args_cli.overwrite:
    raise SystemExit(f"[ERROR]: {output}.hdf5 exists; pass --overwrite to replace it.")
os.makedirs(os.path.dirname(output), exist_ok=True)

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import time

import torch
from IL_mav_carry_ext.mpc import MpcRecorderManagerCfg, dataset_metadata
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

HEARTBEAT_S = 10.0


def print_heartbeat(env, steps, total_steps, elapsed, num_written, episode_steps):
    sim_s = float(env.episode_length_buf.float().mean()) * env.step_dt
    eta_min = elapsed / steps * max(total_steps - steps, 0) / 60
    print(
        f"[REC]   step {steps}/{total_steps} | episode time {sim_s:4.1f}/{episode_steps * env.step_dt:.0f} s | "
        f"{steps * env.num_envs / elapsed:.0f} env-steps/s | "
        f"written {num_written}/{args_cli.num_episodes} | eta {eta_min:.1f} min",
        flush=True,
    )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.pose_command.rebuild = args_cli.rebuild
    env_cfg.recorders = MpcRecorderManagerCfg(
        dataset_export_dir_path=os.path.dirname(output),
        dataset_filename=os.path.basename(output),
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    teacher = env.command_manager.get_term("pose_command").teacher
    recorder = env.recorder_manager

    meta = dataset_metadata(env)
    meta.update(seed=args_cli.seed, num_envs=args_cli.num_envs)
    with open(output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO]: recording to {output}.hdf5", flush=True)

    def written():
        return recorder.exported_successful_episode_count + recorder.exported_failed_episode_count

    episode_steps = int(env.max_episode_length)
    total_steps = math.ceil(args_cli.num_episodes / args_cli.num_envs) * episode_steps
    print(
        f"[INFO]: {args_cli.num_episodes} episodes of {episode_steps} steps on {args_cli.num_envs} envs: "
        f"about {total_steps} env steps; progress every {HEARTBEAT_S:.0f} s",
        flush=True,
    )

    env.reset()
    start = last_beat = time.time()
    steps = 0
    failed_solves = 0
    last_written = 0
    while simulation_app.is_running() and written() < args_cli.num_episodes:
        with torch.inference_mode():
            env.step(teacher.action)
        steps += 1
        failed_solves += int((~teacher.ok).sum())
        now = time.time()
        num_written = written()

        if num_written != last_written:
            last_written = num_written
            print(
                f"[REC] episodes written {num_written}/{args_cli.num_episodes} | "
                f"success {recorder.exported_successful_episode_count / num_written:.0%} | "
                f"failed solves {failed_solves}",
                flush=True,
            )

        if now - last_beat >= HEARTBEAT_S:
            last_beat = now
            print_heartbeat(env, steps, total_steps, now - start, num_written, episode_steps)

    print(
        f"[INFO]: wrote {written()} episodes ({recorder.exported_successful_episode_count} successful) "
        f"in {(time.time() - start) / 60:.1f} min to {output}.hdf5",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
