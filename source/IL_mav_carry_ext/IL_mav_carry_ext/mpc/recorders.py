# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms that write the NMPC teacher's demonstrations to a dataset.

Isaac Lab's `RecorderManager` calls each term inside `env.step` and exports an
env's episode as `data/demo_N` when it resets. These terms record in
`record_pre_step`, after the action arrived and before physics: the policy obs
is still the one the action was computed from, and the teacher's plan was
solved for that same state, so each row is one matching (obs, label) pair.
"""

import numpy as np
import torch
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils.configclass import configclass

from ..tasks.managerbased.mdp.observations import REF_HORIZON_OFFSETS


def teacher_of(env):
    return env.command_manager.get_term("pose_command").teacher


class TeacherHorizonRecorder(RecorderTerm):
    """The teacher's whole plan this step: (num_envs, num_drones, num_nodes, 12), p, v, a, w."""

    def record_pre_step(self):
        return "teacher_horizon", teacher_of(self._env).plan


class TeacherLoadPlanRecorder(RecorderTerm):
    """The payload pose the teacher's plan predicts: (num_envs, num_nodes, 7), p then q (XYZW), env frame."""

    def record_pre_step(self):
        teacher = teacher_of(self._env)
        return "teacher_load_plan", torch.as_tensor(teacher.load_plan, dtype=torch.float32, device=teacher.device)


class TeacherTensionRecorder(RecorderTerm):
    """The cable tensions the teacher's last solve started from: (num_envs, num_drones), N."""

    def record_pre_step(self):
        teacher = teacher_of(self._env)
        return "teacher_tension", torch.as_tensor(teacher.tension, dtype=torch.float32, device=teacher.device)


class RopeForceRecorder(RecorderTerm):
    """Force between the payload and each rope, from the scene's joint wrench sensor: (num_envs, num_drones, 3), N."""

    def record_pre_step(self):
        sensor = self._env.scene["joint_forces"]
        ids = [
            sensor.body_names.index(f"rope_{i + 1}_sphere_joint_0_link_1")
            for i in range(teacher_of(self._env).num_drones)
        ]
        return "rope_force", sensor.data.force.torch[:, ids]


class MpcOkRecorder(RecorderTerm):
    """Whether this step's solve succeeded: (num_envs,) bool."""

    def record_pre_step(self):
        return "mpc_ok", teacher_of(self._env).ok


@configclass
class MpcRecorderManagerCfg(RecorderManagerBaseCfg):
    """Everything one NMPC demonstration dataset holds, per step unless noted.

    - `obs`: the flat policy obs, the student's input; `dataset_metadata`
      says which columns are which term.
    - `teacher_horizon`: the label, each drone's full plan.
    - `teacher_load_plan`: the payload pose that plan predicts, per node.
    - `teacher_tension`: the cable tensions the teacher assumed; `rope_force`: what the sim's ropes pull with.
    - `actions`: the action executed; when the teacher flies, node 1 of the label.
    - `mpc_ok`: whether the label's solve succeeded.
    - `states`: the full scene state after the step; `initial_state` the same,
      once, at the episode's start.

    Every episode is exported, failed ones included; the loader filters.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_obs = PreStepFlatPolicyObservationsRecorderCfg()
    record_teacher_horizon = RecorderTermCfg(class_type=TeacherHorizonRecorder)
    record_teacher_load_plan = RecorderTermCfg(class_type=TeacherLoadPlanRecorder)
    record_teacher_tension = RecorderTermCfg(class_type=TeacherTensionRecorder)
    record_rope_force = RecorderTermCfg(class_type=RopeForceRecorder)
    record_mpc_ok = RecorderTermCfg(class_type=MpcOkRecorder)

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL


def dataset_metadata(env) -> dict:
    """What a loader needs to read a dataset, as JSON-ready builtins.

    The recorded obs is one flat vector; the term layout here is what turns
    its columns back into named terms. The rest pins down what the labels
    mean: node times, frames, and the plant the teacher modelled.
    """
    teacher = teacher_of(env)
    manager = env.observation_manager
    plant = teacher.plant
    return {
        "obs_terms": [
            {"name": name, "dim": int(np.prod(dim))}
            for name, dim in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"], strict=False)
        ],
        "obs_per_drone_terms": "terms named drone_* are drone-major blocks, one per drone",
        "payload_ref_horizon_layout": "per node: dp(3), v(3), a(3), rot6d(6), w(3)",
        "teacher_horizon_layout": "(num_drones, num_nodes, 12): p(3), v(3), a(3), w(3), env frame",
        "action_layout": "per drone: p(3), v(3), a(3), w(3) body-rate reference in the world frame",
        "num_drones": teacher.num_drones,
        "num_nodes": teacher.num_nodes,
        "node_offsets_s": [float(t) for t in REF_HORIZON_OFFSETS],
        "step_dt": float(env.step_dt),
        "episode_length_s": float(env.cfg.episode_length_s),
        "plant": {
            "load_mass": float(plant.load_mass),
            "load_inertia": np.asarray(plant.load_inertia, dtype=float).tolist(),
            "drones": [
                {
                    "mass": float(d.mass),
                    "cable_length": float(d.cable_length),
                    "attach_point": np.asarray(d.attach_point, dtype=float).tolist(),
                }
                for d in plant.drones
            ],
        },
    }
