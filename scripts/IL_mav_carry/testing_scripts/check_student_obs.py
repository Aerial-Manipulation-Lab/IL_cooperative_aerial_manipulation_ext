# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly the NMPC teacher and check the imitation obs terms against what it sees.

Checks, every step and every env whose solve did not fail:
  - `payload_ref_horizon` is the reference window the teacher solves against
    next: same horizon, same samples, same clock, and a rot6d that is the
    payload-relative attitude of each sampled pose.
  - `drone_cable_dir` is unit length and points along the simulated cable,
    and the MPC's attach points sit on the simulated rope attachments.

And, once per second of sim time, how well the teacher flies: the payload's
error to its goal, and each drone's error to the node-1 setpoint it was just
sent. A payload offset the drones share points at the low-level tracking; one
they do not share points at the MPC's model of the geometry.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the imitation obs terms against the NMPC teacher.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=None, help="Env steps to check (100 per second); default one episode.")
parser.add_argument(
    "--episode_length",
    type=float,
    default=None,
    help="Episode length in s, overriding the env cfg, e.g. to see whether the MPC is still converging.",
)
parser.add_argument("--rebuild", action="store_true", default=False, help="Regenerate the acados solver first.")
parser.add_argument("--seed", type=int, default=0, help="Env seed, so runs sample the same goals.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from IL_mav_carry_ext.tasks.managerbased.flycrane.flycrane_env_cfg import ApproachPoseEnvCfg
from IL_mav_carry_ext.tasks.managerbased.mdp_llc import observations as mdp_obs
from isaaclab.envs import ManagerBasedRLEnv
from scipy.spatial.transform import Rotation

ROPE_BODIES = "rope_[1-3]_link"
"""The simulated cables' payload ends, one per drone, in drone order."""


def split_terms(env, obs):
    """The policy obs as {term name: (num_envs, dim) numpy array}."""
    manager = env.unwrapped.observation_manager
    names = manager.active_terms["policy"]
    dims = [int(np.prod(d)) for d in manager.group_obs_term_dim["policy"]]
    cols = np.split(obs["policy"].cpu().numpy(), np.cumsum(dims)[:-1], axis=1)
    return dict(zip(names, cols, strict=True))


def expected_ref_horizon(teacher, i, now):
    """Env i's reference window, rebuilt from the teacher's own objects."""
    policy = teacher.policies[i]
    k = policy.traj.indices_at(policy.horizon.absolute(now))
    load = teacher.measured_load_state(i, now)
    r_rel = Rotation.from_quat(load.q).inv() * Rotation.from_quat(policy.traj.q[k])
    rot6d = r_rel.as_matrix()[..., :2].transpose(0, 2, 1).reshape(len(k), 6)
    return np.concatenate(
        [policy.traj.p[k] - load.p, policy.traj.v[k], policy.traj.a[k], rot6d, policy.traj.w[k]], axis=1
    )


def main():
    env_cfg = ApproachPoseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.actions.low_level_action.control_mode = "geometric"
    env_cfg.seed = args_cli.seed
    if args_cli.episode_length is not None:
        env_cfg.episode_length_s = args_cli.episode_length
    # one full episode by default, so the goal error below covers ramp and settling alike
    num_steps = args_cli.steps or round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    env_cfg.commands.pose_command.rebuild = args_cli.rebuild
    env = ManagerBasedRLEnv(cfg=env_cfg)
    teacher = env.command_manager.get_term("pose_command").teacher
    robot = teacher.robot
    rope_idx = robot.find_bodies(ROPE_BODIES)[0]

    np.testing.assert_array_equal(mdp_obs.REF_HORIZON_OFFSETS, teacher.policies[0].horizon.shooting_nodes)
    print("[CHECK] obs horizon offsets == teacher shooting nodes")

    obs, _ = env.reset()
    worst = {"ref": 0.0, "cable_norm": 0.0, "cable_angle_deg": 0.0, "attach_m": 0.0}
    checked = 0
    failed_solves = 0
    # goal error over time, one row per second of sim time as it happens: still
    # shrinking at the end means the MPC is slow and longer episodes help; flat
    # means it settled with an offset. The signed per-axis mean (payload minus
    # goal) separates a consistent offset (e.g. a z sag, same sign in every env)
    # from a lag, whose direction follows each env's goal. Envs reset when an
    # episode ends, so rows past the first episode mix goals.
    steps_per_s = round(1.0 / env.step_dt)
    print(f"[MPC]   goal error over time, {teacher.num_envs} envs (ramp ends at 3 s)", flush=True)
    print(
        "[MPC]              payload - goal                                                  |  drone - setpoint  |"
        " cable   | MPC's payload z: predicted - measured, horizon end - goal",
        flush=True,
    )
    print(
        "[MPC]     t [s]   |err| mean   max    |  signed mean x      y      z   |  |x|    |y|    |z|  |"
        "  |err|  signed z  | length  |  node 1      end  [cm; cable m]",
        flush=True,
    )
    for step in range(num_steps):
        # the obs in hand is what the next solve acts on, at the env's clock now
        terms = split_terms(env, obs)
        now = env.common_step_counter * env.step_dt
        for i in range(teacher.num_envs):
            got = terms["payload_ref_horizon"][i].reshape(-1, 18)
            worst["ref"] = max(worst["ref"], float(np.abs(got - expected_ref_horizon(teacher, i, now)).max()))

        cable = terms["drone_cable_dir"].reshape(teacher.num_envs, -1, 3)
        worst["cable_norm"] = max(worst["cable_norm"], float(np.abs(np.linalg.norm(cable, axis=-1) - 1).max()))
        state = robot.data.body_com_state_w.torch.cpu().numpy()
        rope = state[:, rope_idx, :3]
        sim_cable = state[:, teacher.falcon_idx, :3] - rope
        sim_cable /= np.linalg.norm(sim_cable, axis=-1, keepdims=True)
        cos = np.clip((cable * sim_cable).sum(-1), -1.0, 1.0)
        worst["cable_angle_deg"] = max(worst["cable_angle_deg"], float(np.degrees(np.arccos(cos)).max()))
        payload = state[:, teacher.load_idx]
        attach = np.stack(
            [payload[:, :3] + Rotation.from_quat(payload[:, 3:7]).apply(d.attach_point) for d in teacher.plant.drones],
            axis=1,
        )
        worst["attach_m"] = max(worst["attach_m"], float(np.linalg.norm(attach - rope, axis=-1).max()))
        if step % steps_per_s == 0:
            e = 100 * np.stack(
                [
                    teacher.measured_load_state(i, now).p - teacher.policies[i].traj.p[-1]
                    for i in range(teacher.num_envs)
                ]
            )
        checked += 1

        failed_solves += int((~teacher.ok).sum())
        # node 1 is the setpoint for one step ahead; the plan is overwritten by the step's solve
        sent = teacher.action.view(teacher.num_envs, -1, 12)[..., :3].cpu().numpy()
        load_plan = teacher.load_plan.copy()
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(teacher.action)

        if step % steps_per_s == 0:
            flown = (
                (robot.data.body_com_state_w.torch[:, teacher.falcon_idx, :3] - teacher.env_origins[:, None])
                .cpu()
                .numpy()
            )
            track = 100 * (flown - sent)
            after = robot.data.body_com_state_w.torch[:, teacher.load_idx].cpu().numpy()
            payload_after = after[:, :3] - teacher.env_origins.cpu().numpy()
            attach_after = payload_after[:, None] + np.stack(
                [Rotation.from_quat(after[:, 3:7]).apply(d.attach_point) for d in teacher.plant.drones], axis=1
            )
            cable_length = np.linalg.norm(flown - attach_after, axis=-1).mean()
            goal_z = np.array([teacher.policies[i].traj.p[-1][2] for i in range(teacher.num_envs)])
            # this step's solve predicted node 1 for exactly the state after the step
            model_z = 100 * (load_plan[:, 1, 2] - payload_after[:, 2]).mean()
            aim_z = 100 * (load_plan[:, -1, 2] - goal_z).mean()
            norm = np.linalg.norm(e, axis=-1)
            m, a = e.mean(0), np.abs(e).mean(0)
            print(
                f"[MPC]     {step * env.step_dt:5.1f}   {norm.mean():7.2f} {norm.max():7.2f}   | "
                f"{m[0]:7.2f} {m[1]:6.2f} {m[2]:6.2f}  | {a[0]:5.2f}  {a[1]:5.2f}  {a[2]:5.2f}  | "
                f"{np.linalg.norm(track, axis=-1).mean():6.2f}  {track[..., 2].mean():7.2f}   | "
                f"{cable_length:6.3f}  | {model_z:7.2f}  {aim_z:7.2f}",
                flush=True,
            )

    print(f"[CHECK] {checked} steps x {teacher.num_envs} envs")
    print(f"[CHECK] payload_ref_horizon  max |obs - teacher|        = {worst['ref']:.2e}")
    print(f"[CHECK] drone_cable_dir      max | |dir| - 1 |           = {worst['cable_norm']:.2e}")
    print(f"[CHECK] drone_cable_dir      max angle to sim cable     = {worst['cable_angle_deg']:.2f} deg")
    print(f"[CHECK] plant attach pts     max distance to sim rope   = {worst['attach_m'] * 100:.2f} cm")
    print(f"[MPC]   failed solves {failed_solves} / {checked * teacher.num_envs}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
