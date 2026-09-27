# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning from the NMPC teacher: student features and model.

Torch only -- nothing here needs the simulator, so training runs without it.
"""

from .features import LABEL_FRAME, PER_DRONE_TERMS, PLAN_DIM, SHARED_TERMS, FeatureSpec
from .model import BCPolicy, load_checkpoint, save_checkpoint

__all__ = [
    "BCPolicy",
    "FeatureSpec",
    "LABEL_FRAME",
    "PER_DRONE_TERMS",
    "PLAN_DIM",
    "SHARED_TERMS",
    "load_checkpoint",
    "save_checkpoint",
]
