# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NMPC teacher plumbing for the flycrane tasks."""

from .teacher import ACADOS_STATUS, FLYCRANE, RAMP_SECONDS, MpcTeacher
from .wrapper import MpcTeacherWrapper

__all__ = ["ACADOS_STATUS", "FLYCRANE", "RAMP_SECONDS", "MpcTeacher", "MpcTeacherWrapper"]
