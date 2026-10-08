# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the uniform per-drone student from scratch with DAgger, labelled by the NMPC teacher.

A round is ROUND_FLIGHT_S of flight over all envs together, as much as one episode of the thesis's
DAgger; the envs carry on across rounds and restart only when their episode ends, staggered so that
each round's flight covers the whole episode once (from round num_envs - 1 on). Per env and step
the student flies if the beta coin says so and its whole plan is within GATE of the teacher's, else
the teacher does; every state is labelled with the teacher's plan. Beta is 1 in round 0, which makes
it behaviour cloning, and 0 from then on, so the gate alone decides: the simplest DAgger (Ross et al.).
Every `--keep_every`-th step goes into the training buffer (steps 10 ms apart are near copies).
After each round the student trains `--epochs` passes over the buffer.

Episodes are recorded to <run dir>/rollouts_<first round>.hdf5. Running the same command again
resumes from <run dir>/last.pt. The student trained in round k is kept as round_<k>.pt; it flies
round k + 1, so that round's numbers are its results.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/dagger.py --headless --task Flycrane-FigureEight-v0 --run_name mlp_gate --model mlp'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger with the NMPC teacher.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Flycrane-ApproachPose-v0, Flycrane-FigureEight-v0 (random), Flycrane-FigureEightA2-v0 or Flycrane-FigureEightA4-v0.",
)
parser.add_argument("--run_name", type=str, required=True, help="Run dir under logs/dagger; an existing run resumes.")
parser.add_argument("--iterations", type=int, default=25, help="Rounds of ROUND_FLIGHT_S flight, then training.")
parser.add_argument("--num_envs", type=int, default=8, help="Flycranes flying in parallel.")
parser.add_argument("--epochs", type=int, default=16, help="Passes over every example so far, per round.")
parser.add_argument("--keep_every", type=int, default=10, help="Train on every Nth step; the recording keeps all.")
parser.add_argument(
    "--model",
    choices=["pin", "mlp"],
    default="pin",
    help="Student network, see build_model; a resumed run keeps its own.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import json
import math
import os
import time

import torch
from IL_mav_carry_ext.imitation import (
    FeatureSpec,
    StudentPolicy,
    bc_loss,
    build_model,
    load_checkpoint,
    load_episodes,
    save_checkpoint,
)
from IL_mav_carry_ext.mpc import MpcRecorderManagerCfg, dataset_metadata
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 256
LR = 1e-3
GATE = 0.03  # the thesis's 0.02 MSE over 20 state columns per node, rescaled to our 12
ROUND_FLIGHT_S = 40.0
LOG_DIR = "logs/dagger"


class Buffer:
    """Every (input, label) example so far, one row per drone and step, on the GPU."""

    def __init__(self, capacity, spec, device):
        self.x = torch.empty(capacity, spec.input_dim, device=device)
        self.y = torch.empty(capacity, spec.label_dim, device=device)
        self.n = 0

    def add(self, inputs, labels):
        x, y = inputs.flatten(0, -2), labels.flatten(0, -2)
        self.x[self.n : self.n + len(x)] = x
        self.y[self.n : self.n + len(y)] = y
        self.n += len(x)

    def batches(self, batch_size):
        """One epoch: every example once, in random order."""
        order = torch.randperm(self.n, device=self.x.device)
        for i in range(0, self.n, batch_size):
            idx = order[i : i + batch_size]
            yield self.x[idx], self.y[idx]


@torch.no_grad()
def mean_loss(model, x, y):
    """bc_loss over many examples, in chunks."""
    return sum(
        bc_loss(model, xb, yb).item() * len(xb) for xb, yb in zip(x.split(4096), y.split(4096), strict=True)
    ) / len(x)


def main():
    run_dir = os.path.abspath(os.path.join(LOG_DIR, args_cli.run_name))
    os.makedirs(run_dir, exist_ok=True)
    last = os.path.join(run_dir, "last.pt")
    ckpt = None
    if os.path.exists(last):
        model, spec, ckpt = load_checkpoint(last)
    start = ckpt["round"] + 1 if ckpt is not None else 0
    rollouts = f"rollouts_{start:03d}"
    # a fresh start ignores leftovers of an attempt that never finished a round
    earlier = sorted(glob.glob(os.path.join(run_dir, "rollouts_*.hdf5"))) if ckpt is not None else []
    earlier = [f for f in earlier if not f.endswith(rollouts + ".hdf5")]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = start
    env_cfg.recorders = MpcRecorderManagerCfg(dataset_export_dir_path=run_dir, dataset_filename=rollouts)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    teacher = env.command_manager.get_term("pose_command").teacher
    meta = dataset_metadata(env)
    with open(os.path.join(run_dir, rollouts + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if ckpt is None:
        spec = FeatureSpec(meta)
        model = build_model(spec, args_cli.model)
    elif spec.meta["obs_terms"] != meta["obs_terms"]:
        raise ValueError(f"{last} was trained on another obs layout than this env has")
    model.to(env.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    if ckpt is not None:
        optimiser.load_state_dict(ckpt["optimizer"])
    student = StudentPolicy(model, spec)

    keep = args_cli.keep_every
    episodes = [(x[::keep], y[::keep]) for x, y in load_episodes(earlier)[0]] if earlier else []
    num_envs, num_drones = env.num_envs, spec.num_drones
    steps = round(ROUND_FLIGHT_S / (num_envs * env.step_dt))
    capacity = sum(x.shape[0] * x.shape[1] for x, _ in episodes)
    capacity += max(args_cli.iterations - start, 0) * math.ceil(steps / keep) * num_envs * num_drones
    buffer = Buffer(capacity, spec, env.device)
    for x, y in episodes:
        buffer.add(x.to(env.device), y.to(env.device))
    del episodes

    writer = SummaryWriter(run_dir)
    print(
        f"[INFO]: rounds {start}..{args_cli.iterations - 1}, {steps} steps on {num_envs} envs each; "
        f"buffer {buffer.n}/{capacity}; logging to {run_dir}",
        flush=True,
    )

    obs, _ = env.reset()
    # cut each env's first episode short by a different 1/num_envs, so from then on they fly
    # different parts of it and every round holds the whole episode once
    env.episode_length_buf[:] = torch.arange(num_envs, device=env.device) * int(env.max_episode_length) // num_envs
    for rnd in range(start, args_cli.iterations):
        if not simulation_app.is_running():
            break
        round_start = time.time()
        beta = 1.0 if rnd == 0 else 0.0
        vel_err = acc_err = err_n = failed = ended_early = timeouts = flown = 0
        ended_by = {}
        fresh = buffer.n
        with torch.no_grad():
            for step in range(steps):
                x, ok = obs["policy"], teacher.ok
                failed += int((~ok).sum())
                if step % keep == 0:
                    buffer.add(spec.build_inputs(x[ok]), spec.plan_to_label(teacher.plan[ok], x[ok]))
                action = teacher.action
                if rnd > 0:
                    plan = student.plan(x)
                    gap = plan - teacher.plan
                    node1 = gap[ok][:, :, 1]  # what the drones are sent: node 1's velocity and acceleration
                    vel_err += float(node1[..., 3:6].norm(dim=-1).mean(-1).sum())
                    acc_err += float(node1[..., 6:9].norm(dim=-1).mean(-1).sum())
                    err_n += int(ok.sum())
                    close = gap.pow(2).mean((1, 2, 3)) < GATE
                    student_flies = (torch.rand(num_envs, device=env.device) >= beta) & close & ok
                    flown += int(student_flies.sum())
                    action = torch.where(student_flies[:, None], spec.plan_to_action(plan), teacher.action)
                obs, _, terminated, truncated, _ = env.step(action)
                ended_early += int(terminated.sum())
                timeouts += int(truncated.sum())
                if bool((terminated | truncated).any()):
                    for name in env.termination_manager.active_terms:
                        ended_by[name] = ended_by.get(name, 0) + int(env.termination_manager.get_term(name).sum())

        # the student's loss on the states it just flew, before it trains on them
        new_loss = (
            mean_loss(model, buffer.x[fresh : buffer.n], buffer.y[fresh : buffer.n])
            if err_n and buffer.n > fresh
            else float("nan")
        )
        loss_sum = 0.0
        for _ in range(args_cli.epochs):
            for xb, yb in buffer.batches(BATCH_SIZE):
                loss = bc_loss(model, xb, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                loss_sum += loss.item() * len(xb)

        metrics = {
            "beta": beta,
            "vel_err_mps": vel_err / err_n if err_n else float("nan"),
            "acc_err_mps2": acc_err / err_n if err_n else float("nan"),
            "new_loss": new_loss,
            "student_share": flown / (steps * num_envs),
            "success": timeouts,
            "ended_early": ended_early,
            "failed_solves": failed,
            "loss": loss_sum / (args_cli.epochs * buffer.n),
            "buffer": buffer.n,
            **{f"ended_by/{name}": count for name, count in ended_by.items()},
        }
        for key, value in metrics.items():
            writer.add_scalar(key, value, rnd)
        print(
            f"[ROUND {rnd:3d}/{args_cli.iterations}] beta {beta:.2f} | student flew {100 * flown / (steps * num_envs):.0f}% | "
            f"err vel {metrics['vel_err_mps']:.3f} m/s "
            f"acc {metrics['acc_err_mps2']:.3f} m/s2 | new loss {new_loss:.3f} loss {metrics['loss']:.3f} | "
            f"success {timeouts} | ended early {ended_early} | failed solves {failed} | "
            f"buffer {buffer.n} | {(time.time() - round_start) / 60:.1f} min | ended by "
            + (", ".join(f"{name} {count}" for name, count in ended_by.items() if count) or "-"),
            flush=True,
        )
        save_checkpoint(os.path.join(run_dir, f"round_{rnd:03d}.pt"), model, spec, round=rnd, metrics=metrics)
        save_checkpoint(last, model, spec, round=rnd, metrics=metrics, optimizer=optimiser.state_dict())

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
