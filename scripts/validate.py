# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly a trained student, or the NMPC teacher with --teacher, on every reference in VALIDATION at once.

One env, Flycrane-Validation-v0, with an env per flight: env i flies the i-th reference, round robin, from its own
random start, so the references fly side by side and the teacher's solves overlap. The random ones draw a new walk
or figure eight per flight. Reports play.py's metrics per reference.

Example, from the repo root:

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \\
        IL_cooperative_aerial_manipulation_ext/scripts/validate.py \\
        --headless --checkpoint logs/dagger/<run_name>/round_024.pt'
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Validate a student or the teacher on every test reference."
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Checkpoint from scripts/train.py or dagger.py."
)
parser.add_argument(
    "--teacher", action="store_true", default=False, help="Fly the NMPC teacher, as the baseline."
)
parser.add_argument(
    "--flights", type=int, default=1, help="Flights per reference, each from its own random start."
)
parser.add_argument(
    "--seed", type=int, default=0, help="The same seed gives the same starts and random draws."
)
# a plant unlike the model: the sim's payload changed, every controller keeping its model
parser.add_argument(
    "--payload_mass_scale", type=float, default=1.0, help="The sim payload's mass, times this."
)
parser.add_argument(
    "--payload_com_offset", type=float, default=0.0, help="The sim payload's COM moved along x, m."
)
# a model unlike the plant: the teacher and every model-based input assume this, the sim unchanged
parser.add_argument(
    "--model_cable_length", type=float, default=None, help="The model's cable length, m."
)
parser.add_argument(
    "--model_attach_offset",
    type=float,
    default=0.0,
    help="The model's attach points moved along x, m.",
)
# disturbances: gusts on the payload and noisy measurements
parser.add_argument(
    "--payload_force",
    type=float,
    default=0.0,
    help="Gusts on the payload, up to this force per axis in N, redrawn every 1 to 3 s.",
)
parser.add_argument(
    "--payload_torque",
    type=float,
    default=0.0,
    help="Gusts on the payload, up to this torque per axis in N m, redrawn with the force.",
)
parser.add_argument(
    "--noise_scale",
    type=float,
    default=0.0,
    help="Noise on what the student and the pilot measure, times the configured spreads.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.teacher and args_cli.checkpoint is None:
    parser.error("--checkpoint is required unless --teacher")

"""Launch Isaac Sim Simulator first."""

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import replace

import IL_mav_carry_ext.tasks.managerbased.mdp as mdp
import numpy as np
import torch
from IL_mav_carry_ext.imitation import FeatureSpec, load_checkpoint
from IL_mav_carry_ext.plants import FLYCRANE_SIM
from IL_mav_carry_ext.reporting import TextReporter
from IL_mav_carry_ext.tasks.managerbased.flycrane.flycrane_env_cfg import VALIDATION
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Flycrane-Validation-v0"


def main():
    report = TextReporter("validate")
    student, spec = (
        (None, None)
        if args_cli.teacher
        else load_checkpoint(args_cli.checkpoint, args_cli.device)[:2]
    )
    names = list(VALIDATION)
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=len(names) * args_cli.flights)
    env_cfg.seed = args_cli.seed
    # a failed teacher solve says nothing about the student's flight
    env_cfg.terminations.teacher_solve_failed = None
    env_cfg.commands.pose_command.solve_teacher = args_cli.teacher
    payload = SceneEntityCfg("robot", body_names="load_link")
    if args_cli.payload_mass_scale != 1.0:
        env_cfg.events.payload_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": payload,
                "mass_distribution_params": (
                    args_cli.payload_mass_scale,
                    args_cli.payload_mass_scale,
                ),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
    if args_cli.payload_com_offset:
        offset = (args_cli.payload_com_offset, args_cli.payload_com_offset)
        env_cfg.events.payload_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={"asset_cfg": payload, "com_range": {"x": offset}},
        )
    if args_cli.payload_force or args_cli.payload_torque:
        env_cfg.events.payload_gusts = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "asset_cfg": payload,
                "force_range": (-args_cli.payload_force, args_cli.payload_force),
                "torque_range": (-args_cli.payload_torque, args_cli.payload_torque),
            },
        )
    if args_cli.noise_scale:
        # the student's measured inputs carry their configured noise, scaled; references carry none
        env_cfg.observations.policy.enable_corruption = True
        for term in vars(env_cfg.observations.policy).values():
            if getattr(term, "noise", None) is not None:
                term.noise.std = term.noise.std * args_cli.noise_scale
        env_cfg.actions.low_level_action.state_noise_scale = args_cli.noise_scale
    if args_cli.model_cable_length is not None or args_cli.model_attach_offset:
        cables = tuple(
            replace(
                c,
                length=args_cli.model_cable_length or c.length,
                attach_point=c.attach_point + np.array([args_cli.model_attach_offset, 0.0, 0.0]),
            )
            for c in FLYCRANE_SIM.cables
        )
        env_cfg.commands.pose_command.model = replace(FLYCRANE_SIM, cables=cables)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    if spec is not None and spec != FeatureSpec.from_env(env):
        raise ValueError(f"{args_cli.checkpoint} was trained on other rows than {TASK} gives")
    command, terminations = env.command_manager.get_term("pose_command"), env.termination_manager

    n = env.num_envs
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    flights = []  # (reference, its RMSEs, the term that ended it), one per env as it ends
    obs, _ = env.reset()
    while simulation_app.is_running() and not done.all():
        # each env's RMSE so far, read before the step whose reset zeroes it: it misses only that last step
        before = {name: value.clone() for name, value in command.metrics.items()}
        with torch.inference_mode():
            action = obs["label"] if student is None else student.predict(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(action.flatten(1))
        ended = (terminated | truncated) & ~done
        for i in ended.nonzero().flatten().tolist():
            ended_by = next(t for t in terminations.active_terms if terminations.get_term(t)[i])
            flights.append(
                (names[i % len(names)], {k: float(v[i]) for k, v in before.items()}, ended_by)
            )
        done |= ended
    env.close()

    for name in names:
        mine = [(numbers, by) for reference, numbers, by in flights if reference == name]
        completed = [numbers for numbers, by in mine if by == "success"]
        result = {"completed": len(completed), "flights": len(mine)}
        if completed:
            result |= {k: float(np.mean([c[k] for c in completed])) for k in completed[0]}
        for by in sorted({by for _, by in mine}):
            result[f"ended_by_{by}_share"] = sum(b == by for _, b in mine) / len(mine)
        report.metrics(f"result {name}", result)


if __name__ == "__main__":
    main()
    simulation_app.close()
