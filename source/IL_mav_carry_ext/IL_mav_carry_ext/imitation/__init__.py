# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning from the NMPC teacher: the student models and their training.

Torch only -- nothing here needs the simulator, so training runs without it.
"""

from .features import PLAN_DIM, input_dim, label_dim
from .model import BCPolicy, build_model, load_checkpoint, save_checkpoint
from .training_utils import bc_loss, load_episodes

__all__ = [
    "PLAN_DIM",
    "BCPolicy",
    "bc_loss",
    "build_model",
    "input_dim",
    "label_dim",
    "load_checkpoint",
    "load_episodes",
    "save_checkpoint",
]
