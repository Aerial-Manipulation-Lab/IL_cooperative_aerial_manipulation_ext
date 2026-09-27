# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms that write the NMPC teacher's demonstrations to a dataset.

Isaac Lab's `RecorderManager` calls each term at fixed points inside
`env.step` / `env.reset`, appends what it returns to each env's running
episode, and exports that episode as `data/demo_N` when the env resets. Every
term here records in `record_pre_step`: after the step's action arrived,
before physics. At that point the policy obs is still the one the action was
computed from, and `MpcTeacherWrapper` has just left the step's teacher output
on the env -- so each recorded row is one matching (obs, label) pair.

Only usable under `MpcTeacherWrapper`, which is what sets the attributes these
terms read; the recording script switches it on with
``env_cfg.recorders = MpcRecorderManagerCfg()``.
"""

import numpy as np

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

from ..tasks.managerbased.mdp_llc.observations import REF_HORIZON_OFFSETS


class TeacherActionRecorder(RecorderTerm):
    """The teacher's action this step, in the action term's format: (num_envs, action_dim)."""

    def record_pre_step(self):
        return "teacher_action", self._env.teacher_action


class TeacherHorizonRecorder(RecorderTerm):
    """The teacher's whole plan this step: (num_envs, num_drones, num_nodes, 12), p, v, a, w."""

    def record_pre_step(self):
        return "teacher_horizon", self._env.teacher_horizon


class MpcOkRecorder(RecorderTerm):
    """Whether this step's solve succeeded: (num_envs,) bool."""

    def record_pre_step(self):
        return "mpc_ok", self._env.mpc_ok


class TeacherExecutedRecorder(RecorderTerm):
    """Whether the teacher's action was the one applied this step: (num_envs,) bool."""

    def record_pre_step(self):
        return "teacher_executed", self._env.teacher_executed


@configclass
class MpcRecorderManagerCfg(RecorderManagerBaseCfg):
    """Everything one NMPC demonstration dataset holds, per step unless noted.

    - `obs`: the flat policy obs, the student's input; `dataset_metadata`
      says which columns are which term.
    - `teacher_horizon`: the label, each drone's full plan.
    - `teacher_action`: the label's node 1, as the action term takes it.
    - `actions`: the action actually executed (differs from the label once a
      student flies).
    - `mpc_ok`, `teacher_executed`: per-step flags for filtering.
    - `states`: the full scene state after the step, for clean drone states
      and replay; `initial_state` the same, once, at the episode's start.

    Every episode is exported, failed ones included; the loader filters on
    the `success` attribute and `mpc_ok`, so no decision is baked in here.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_obs = PreStepFlatPolicyObservationsRecorderCfg()
    record_teacher_action = RecorderTermCfg(class_type=TeacherActionRecorder)
    record_teacher_horizon = RecorderTermCfg(class_type=TeacherHorizonRecorder)
    record_mpc_ok = RecorderTermCfg(class_type=MpcOkRecorder)
    record_teacher_executed = RecorderTermCfg(class_type=TeacherExecutedRecorder)

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL


def dataset_metadata(env, teacher) -> dict:
    """What a loader needs to read a dataset, as JSON-ready builtins.

    The recorded obs is one flat vector; the term layout here is what turns
    its columns back into named terms. The rest pins down what the labels
    mean: node times, frames, and the plant the teacher modelled.
    """
    base = env.unwrapped
    manager = base.observation_manager
    plant = teacher.plant
    return {
        "obs_terms": [
            {"name": name, "dim": int(np.prod(dim))}
            for name, dim in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"])
        ],
        "obs_per_drone_terms": "terms named drone_* are drone-major blocks, one per drone",
        "payload_ref_horizon_layout": "per node: dp(3), v(3), a(3), rot6d(6), w(3)",
        "teacher_horizon_layout": "(num_drones, num_nodes, 12): p(3), v(3), a(3), w(3), env frame",
        "action_layout": "per drone: p(3), v(3), a(3), zeros(3)",
        "num_drones": teacher.num_drones,
        "num_nodes": teacher.num_nodes,
        "node_offsets_s": [float(t) for t in REF_HORIZON_OFFSETS],
        "step_dt": float(base.step_dt),
        "episode_length_s": float(base.cfg.episode_length_s),
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
