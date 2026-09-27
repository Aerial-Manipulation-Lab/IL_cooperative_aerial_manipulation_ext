"""Roll out a trained student, optionally with a viewport or video recording.

The student flies (`--beta 0`, the default) while the NMPC teacher keeps
solving alongside, so every step also yields the label for the state the
*student* put the system in. That gives two kinds of numbers:

  - outcome: how many episodes run to the end without crashing, and how close
    to the goal the payload ends up;
  - label error: how far the student's executed setpoint is from the one the
    teacher would have commanded in the same state. Growing along an episode,
    it is the drift off the training distribution that DAgger addresses.

`--beta 1` flies the teacher through the very same script: the baseline the
student's outcome is compared against. A failed teacher solve does not reset
the episode here (it says nothing about the student); it is only counted.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/play.py \
        --headless --checkpoint logs/bc/<run>/best.pt --num_episodes 20'
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out a trained student next to the NMPC teacher.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint from scripts/train.py.")
parser.add_argument("--num_envs", type=int, default=4, help="Flycranes flying in parallel.")
parser.add_argument("--num_episodes", type=int, default=None,
                    help="Stop once this many episodes have ended; default one per env, i.e. a single round.")
parser.add_argument("--beta", type=float, default=0.0,
                    help="Probability the teacher's action is executed per step: 0 student, 1 teacher baseline.")
parser.add_argument("--seed", type=int, default=0, help="Env seed; the same seed samples the same goals.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the first steps.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--rebuild", action="store_true", default=False, help="Regenerate the acados solver first.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_episodes is None:
    args_cli.num_episodes = args_cli.num_envs
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from IL_mav_carry_ext.imitation import load_checkpoint
from IL_mav_carry_ext.mpc import MpcTeacherWrapper
from IL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv

HEARTBEAT_S = 10.0
"""Print a progress line at least this often, in wall-clock seconds: episodes all
last the same, so they end in batches minutes apart."""


class StudentPolicy:
    """Checkpoint -> callable from the policy obs to the action term's action."""

    def __init__(self, path, device):
        self.model, self.spec, ckpt = load_checkpoint(path, device=device)
        print(f"[INFO]: student from {path} (epoch {ckpt['epoch']}, "
              f"val node-1 pos {ckpt['metrics']['val/node1_pos_cm']:.2f} cm)")

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        label = self.model.predict(self.spec.build_inputs(obs))
        return self.spec.plan_to_action(self.spec.label_to_plan(label, obs))


def main():
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # camera: track the flycrane instead of staring at the world origin
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (14, 14, 14)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    base = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    teacher_env = MpcTeacherWrapper(
        base, beta=args_cli.beta, reset_on_failed_solve=False, rebuild=args_cli.rebuild, verbose=False
    )
    env = teacher_env
    if args_cli.video:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        env = gym.wrappers.RecordVideo(
            teacher_env,
            video_folder="./videos",
            name_prefix=f"play_{name}_beta{args_cli.beta:g}_{run_id}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    student = StudentPolicy(args_cli.checkpoint, base.device)
    num_envs, num_drones = base.num_envs, teacher_env.teacher.num_drones

    # per-env running sums for the episode in flight
    label_err_sum = np.zeros(num_envs)
    label_err_steps = np.zeros(num_envs)
    # one row per ended episode: (success, final goal error m, mean label error m, steps)
    ended = []
    failed_solves = 0

    obs, _ = env.reset()
    start = last_beat = time.time()
    steps = 0
    while simulation_app.is_running() and len(ended) < args_cli.num_episodes:
        with torch.inference_mode():
            action = student(obs["policy"])
            obs, _, terminated, truncated, info = env.step(action)

        # the teacher solved the same state the student acted on; compare the
        # node-1 positions each drone was sent to, where the solve is trustworthy
        sent = action.view(num_envs, num_drones, 12)[..., :3]
        label = base.teacher_action.view(num_envs, num_drones, 12)[..., :3]
        err = (sent - label).norm(dim=-1).mean(dim=-1).cpu().numpy()
        ok = base.mpc_ok.cpu().numpy()
        label_err_sum[ok] += err[ok]
        label_err_steps[ok] += 1
        failed_solves += len(info["mpc_failed"])
        steps += 1

        if time.time() - last_beat >= HEARTBEAT_S:
            last_beat = time.time()
            in_flight = label_err_sum / np.maximum(label_err_steps, 1)
            print(
                f"[PLAY]  episode time {float(base.episode_length_buf.float().mean()) * base.step_dt:4.1f}/"
                f"{base.max_episode_length * base.step_dt:.0f} s | "
                f"{steps * num_envs / (last_beat - start):.0f} env-steps/s | "
                f"ended {len(ended)}/{args_cli.num_episodes} | "
                f"label err so far {in_flight.mean() * 100:.2f} cm",
                flush=True,
            )

        done = (terminated | truncated).cpu().numpy()
        for i in np.flatnonzero(done):
            if len(ended) == args_cli.num_episodes:
                break
            # the time-out is the `success` term; anything else ended it early
            success = bool(truncated[i])
            mean_err = label_err_sum[i] / max(label_err_steps[i], 1)
            # measured at this step's solve, one step before the episode ended
            ended.append((success, float(info["mpc_pos_err"][i]), mean_err, int(label_err_steps[i])))
            print(
                f"[EP {len(ended):3d}] env {i}  {'success' if success else 'crash  '}  "
                f"goal err {ended[-1][1] * 100:6.2f} cm  label err {mean_err * 100:6.2f} cm",
                flush=True,
            )
            label_err_sum[i] = label_err_steps[i] = 0

        if args_cli.video and teacher_env.count == args_cli.video_length:
            break

    if ended:
        success, goal_err, label_err, _ = (np.asarray(col) for col in zip(*ended))
        who = "student" if args_cli.beta == 0 else ("teacher" if args_cli.beta == 1 else f"beta={args_cli.beta:g}")
        print(f"\n[RESULT] {who}, {len(ended)} episodes")
        print(f"  success            {success.mean():.0%} ({int(success.sum())}/{len(ended)})")
        if success.any():
            print(f"  final goal error   mean {goal_err[success].mean() * 100:.2f} cm, "
                  f"max {goal_err[success].max() * 100:.2f} cm (successful episodes)")
        print(f"  label error        mean {label_err.mean() * 100:.2f} cm (node-1 position vs teacher)")
        print(f"  failed solves      {failed_solves}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
