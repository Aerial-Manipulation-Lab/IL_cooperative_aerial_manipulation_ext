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
import numpy as np
import torch
from IL_mav_carry_ext.imitation import StudentPolicy
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
    student = None if args_cli.teacher else StudentPolicy.load(args_cli.checkpoint, args_cli.device)
    results = []
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
        command = env.command_manager.get_term("pose_command")
        teacher = command.teacher

        n, dt = env.num_envs, env.step_dt
        every = round(0.1 / dt)
        pairs = np.triu_indices(teacher.num_drones, 1)
        origins = env.scene.env_origins.cpu().numpy()
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
                now = env.common_step_counter * dt
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
        env.close()
        SimulationContext.clear_instance()

        pos = np.sqrt(pos_sq / np.maximum(samples, 1))
        ang = np.sqrt(ang_sq / np.maximum(samples, 1))
        print()
        for i in range(n):
            print(
                f"  env {i}: {'completed' if completed[i] else 'crashed'} after {flown[i]:.1f} s | "
                f"position RMSE {pos[i]:.3f} m | attitude RMSE {ang[i]:.2f} deg | closest drones {closest[i]:.2f} m"
            )
        c = completed
        results.append(
            f"[RESULT] {task}: {int(c.sum())}/{n} completed | position RMSE {pos[c].mean():.3f} m | "
            f"attitude RMSE {ang[c].mean():.2f} deg (means over completed flights) | closest drones {closest.min():.2f} m"
        )
        print(results[-1], flush=True)

    print("\n" + "\n".join(results), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
