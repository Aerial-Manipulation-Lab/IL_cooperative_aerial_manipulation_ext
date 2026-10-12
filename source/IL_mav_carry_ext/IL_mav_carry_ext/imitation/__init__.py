# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning from the NMPC teacher: student features and model.

Torch only -- nothing here needs the simulator, so training runs without it.
"""

from .features import LABEL_FRAMES, PLAN_DIM, FeatureSpec
from .model import ThesisPolicy, build_model, load_checkpoint, save_checkpoint
from .student import StudentPolicy
from .training_utils import bc_loss, load_episodes

__all__ = [
    "LABEL_FRAMES",
    "PLAN_DIM",
    "ThesisPolicy",
    "FeatureSpec",
    "bc_loss",
    "build_model",
    "load_episodes",
    "load_checkpoint",
    "save_checkpoint",
    "StudentPolicy",
]
