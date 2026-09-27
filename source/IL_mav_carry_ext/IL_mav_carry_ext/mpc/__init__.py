# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher plumbing for the flycrane tasks."""

from .recorders import MpcRecorderManagerCfg, dataset_metadata
from .teacher import FLYCRANE, FLYCRANE_SIM, MpcTeacher
from .wrapper import MpcTeacherWrapper

__all__ = [
    "FLYCRANE",
    "FLYCRANE_SIM",
    "MpcRecorderManagerCfg",
    "MpcTeacher",
    "MpcTeacherWrapper",
    "dataset_metadata",
]
