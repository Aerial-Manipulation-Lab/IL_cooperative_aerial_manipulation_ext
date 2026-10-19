# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning from the NMPC teacher: the student and what trains and flies it.

Torch and Isaac Lab's configclass only -- nothing here needs the simulator, so training runs without it.
"""

from .algorithms import BehaviourCloning, Dagger, DaggerCfg
from .features import PLAN_DIM, PLAN_PARTS, FeatureSpec
from .models import BCPolicy
from .utils import Buffer, bc_loss, build_model, evaluate, fit, load_checkpoint, load_episodes, save_checkpoint

__all__ = [
    "PLAN_DIM",
    "PLAN_PARTS",
    "BCPolicy",
    "BehaviourCloning",
    "Buffer",
    "Dagger",
    "DaggerCfg",
    "FeatureSpec",
    "bc_loss",
    "build_model",
    "evaluate",
    "fit",
    "load_checkpoint",
    "load_episodes",
    "save_checkpoint",
]
