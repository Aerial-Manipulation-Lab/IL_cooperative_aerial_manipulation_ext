# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record NMPC teacher demonstrations to an HDF5 dataset for imitation learning.

Flies the teacher in every env and lets Isaac Lab's recorder write each finished
episode to `<output>.hdf5` (see `IL_mav_carry_ext.mpc.recorders` for what one
holds), until `--num_episodes` have been written. Next to it goes
`<output>.meta.json`, what a loader needs to read the file.

Every episode is written, failed ones too; filter on the `success` attribute and
the per-step `mpc_ok` when loading. The file is flushed after every episode, so
an interrupted run keeps everything recorded up to that point.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/record.py \
        --headless --num_envs 8 --num_episodes 200 --output datasets/mpc_demos'
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record NMPC teacher demonstrations.")
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

# refuse before the (slow) sim launch: the recorder opens the file for writing,
# which would silently replace a finished dataset
import os

output = os.path.abspath(os.path.splitext(args_cli.output)[0])
if os.path.exists(output + ".hdf5") and not args_cli.overwrite:
    raise SystemExit(f"[ERROR]: {output}.hdf5 exists; pass --overwrite to replace it.")
os.makedirs(os.path.dirname(output), exist_ok=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import time

import torch
from IL_mav_carry_ext.mpc import (
    MpcRecorderManagerCfg,
    MpcTeacherWrapper,
    dataset_metadata,
)
from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc
from isaaclab.envs import ManagerBasedRLEnv

HEARTBEAT_S = 10.0
"""Print a progress line at least this often, in wall-clock seconds. Every env
runs episodes of the same length, so episodes finish in batches minutes apart;
the heartbeat reports progress through the batch in flight in between."""


def main():
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = MpcRecorderManagerCfg(
        dataset_export_dir_path=os.path.dirname(output),
        dataset_filename=os.path.basename(output),
    )

    env = MpcTeacherWrapper(ManagerBasedRLEnv(cfg=env_cfg), rebuild=args_cli.rebuild, verbose=False)
    recorder = env.unwrapped.recorder_manager

    # written before flying, so a run that dies halfway still leaves a readable dataset
    meta = dataset_metadata(env, env.teacher)
    meta.update(seed=args_cli.seed, num_envs=args_cli.num_envs)
    with open(output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO]: recording to {output}.hdf5", flush=True)

    def written():
        return recorder.exported_successful_episode_count + recorder.exported_failed_episode_count

    base = env.unwrapped
    episode_steps = int(base.max_episode_length)
    # a lower bound: episodes that crash early end sooner, so the eta only shrinks
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
            _, _, _, _, info = env.step(None)
        steps += 1
        failed_solves += len(info["mpc_failed"])
        now = time.time()

        if written() != last_written:
            last_written = written()
            print(
                f"[REC] episodes written {last_written}/{args_cli.num_episodes} | "
                f"success {recorder.exported_successful_episode_count / last_written:.0%} | "
                f"failed solves {failed_solves}",
                flush=True,
            )

        if now - last_beat >= HEARTBEAT_S:
            last_beat = now
            elapsed = now - start
            sim_s = float(base.episode_length_buf.float().mean()) * base.step_dt
            speed = steps * base.step_dt / elapsed  # sim seconds per wall second, per env
            eta_min = elapsed / steps * max(total_steps - steps, 0) / 60
            print(
                f"[REC]   step {steps}/{total_steps} | episode time {sim_s:4.1f}/{episode_steps * base.step_dt:.0f} s | "
                f"{steps * args_cli.num_envs / elapsed:.0f} env-steps/s ({speed:.2f}x real time) | "
                f"written {written()}/{args_cli.num_episodes} | eta {eta_min:.1f} min",
                flush=True,
            )

    # episodes still in flight are dropped: the recorder only exports on reset
    print(
        f"[INFO]: wrote {written()} episodes ({recorder.exported_successful_episode_count} successful) "
        f"in {(time.time() - start) / 60:.1f} min to {output}.hdf5",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
