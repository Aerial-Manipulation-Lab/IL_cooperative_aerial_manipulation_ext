# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher plumbing for the flycrane tasks."""

from ..plants import FLYCRANE, FLYCRANE_SIM
from .recorders import MpcRecorderManagerCfg
from .teacher import MpcTeacher

__all__ = [
    "FLYCRANE",
    "FLYCRANE_SIM",
    "MpcRecorderManagerCfg",
    "MpcTeacher",
]
