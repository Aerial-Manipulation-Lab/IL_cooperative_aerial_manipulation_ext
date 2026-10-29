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


def teacher_of(env):
    return env.command_manager.get_term("pose_command").teacher


class LabelRecorder(RecorderTerm):
    """The label group: each drone's teacher plan in its own frame, (num_envs, num_drones, num_nodes * 12)."""

    def record_pre_step(self):
        return "label", self._env.obs_buf["label"]


class TeacherLoadPlanRecorder(RecorderTerm):
    """The payload pose the teacher's plan predicts: (num_envs, num_nodes, 7), p then q (XYZW), env frame."""

    def record_pre_step(self):
        teacher = teacher_of(self._env)
        poses = [np.concatenate([p.load.position, p.load.orientation], -1) for p in teacher.plans]
        return "teacher_load_plan", torch.as_tensor(
            np.stack(poses), dtype=torch.float32, device=teacher.device
        )


class TeacherTensionRecorder(RecorderTerm):
    """The cable tensions the teacher's last solve started from: (num_envs, num_drones), N."""

    def record_pre_step(self):
        teacher = teacher_of(self._env)
        tensions = [[c.tension[0] for c in p.cables] for p in teacher.plans]
        return "teacher_tension", torch.tensor(tensions, dtype=torch.float32, device=teacher.device)


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

    - `obs`: the policy group, the student's input, one row per drone;
      the `FeatureSpec` in the dataset's meta.json says which columns are which term.
    - `label`: the label group, each drone's teacher plan in its own frame.
    - `teacher_load_plan`: the payload pose that plan predicts, per node.
    - `teacher_tension`: the cable tensions the teacher assumed; `rope_force`: what the sim's ropes pull with.
    - `actions`: the action executed; when the teacher flies, the label.
    - `mpc_ok`: whether the label's solve succeeded.
    - `states`: the full scene state after the step; `initial_state` the same,
      once, at the episode's start.

    Every episode is exported, failed ones included; the loader filters.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_obs = PreStepFlatPolicyObservationsRecorderCfg()
    record_label = RecorderTermCfg(class_type=LabelRecorder)
    record_teacher_load_plan = RecorderTermCfg(class_type=TeacherLoadPlanRecorder)
    record_teacher_tension = RecorderTermCfg(class_type=TeacherTensionRecorder)
    record_rope_force = RecorderTermCfg(class_type=RopeForceRecorder)
    record_mpc_ok = RecorderTermCfg(class_type=MpcOkRecorder)

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
