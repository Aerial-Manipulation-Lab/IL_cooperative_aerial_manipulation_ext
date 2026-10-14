# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning from the NMPC teacher: the student and what trains and flies it.

Torch only -- nothing here needs the simulator, so training runs without it.
"""

from .algorithms import BehaviourCloning, Dagger
from .model import BCPolicy
from .utils import PLAN_DIM, Buffer, bc_loss, build_model, evaluate, fit, input_dim, label_dim, load_checkpoint, load_episodes, save_checkpoint

__all__ = [
    "PLAN_DIM",
    "BCPolicy",
    "BehaviourCloning",
    "Buffer",
    "Dagger",
    "bc_loss",
    "build_model",
    "evaluate",
    "fit",
    "input_dim",
    "label_dim",
    "load_checkpoint",
    "load_episodes",
    "save_checkpoint",
]
