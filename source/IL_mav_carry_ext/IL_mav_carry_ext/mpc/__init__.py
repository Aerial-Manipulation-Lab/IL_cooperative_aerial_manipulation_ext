# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher plumbing for the flycrane tasks."""

from .teacher import FLYCRANE, MpcTeacher
from .wrapper import MpcTeacherWrapper

__all__ = ["FLYCRANE", "MpcTeacher", "MpcTeacherWrapper"]
