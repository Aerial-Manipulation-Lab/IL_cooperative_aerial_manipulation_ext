# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.utils.configclass import (
    configclass,  # Imported explicitly, since isaaclab.utils lazy-exports this name.
)
from python_mpc_cusadi import FigureEight, LoadState, SplinePath
from python_mpc_cusadi.signals.specs import SPLINE_DEGREE

WALK_INTERVAL = 0.95
"""Time between a random walk's control points, per sqrt(size / acceleration).

Calibrated offline so that, over many walks, the median peak acceleration is the one drawn.
"""

WALK_STEP = 1.22
"""A random walk's longest step between control points, per size.

Calibrated offline so that, over many walks, the median peak speed is sqrt(size * acceleration).
"""


@configclass
class RandomFigureEight:
    """A figure eight with a random size, speed and heading rate, drawn anew at every reset."""

    # ranges, as (min, max)
    amplitude: tuple[float, float] = MISSING  # half the x-span, m
    acceleration: tuple[float, float] = MISSING  # peak at full speed, sets the period, m/s^2
    yaw_rate: tuple[float, float] = MISSING  # rad/s

    def draw(self, start: LoadState, episode: float) -> FigureEight:
        """Draw one figure eight from the ranges.

        Args:
            start: The payload's state at the reset, unused here.
            episode: Length of the episode, in s, unused here.

        Returns:
            The figure eight, to be placed at the payload by the command.
        """
        amplitude, acceleration, yaw_rate = (
            np.random.uniform(*bounds)
            for bounds in (self.amplitude, self.acceleration, self.yaw_rate)
        )
        # A FigureEight's peak acceleration at full speed is 17/8 amplitude (2 pi / period)^2.
        period = 2 * np.pi * np.sqrt(17 / 8 * amplitude / acceleration)
        return FigureEight(amplitude=amplitude, period=period, yaw_rate=yaw_rate)


@configclass
class RandomWalk:
    """A random walk through space, flown from rest to rest, drawn anew at every reset.

    Each draw picks how much room the walk's turns take (its size, speed^2 / acceleration), its
    peak acceleration and its peak heading rate. The size sets the step between control points,
    and the size and acceleration together set the time between them, so the walk peaks at about
    the speed and acceleration drawn. The course keeps its direction with a random turn at each
    step, mostly level, and turns back when it meets a wall. A smooth spline is then fitted
    through the control points, which never leaves them, so it stays within the walls too.

    The walk is drawn around the origin at heading 0, and the command then moves it to the
    payload and turns it by the payload's yaw, like any shape.
    """

    # ranges, as (min, max)
    size: tuple[float, float] = MISSING  # room the walk's turns take, speed^2 / acceleration, m
    acceleration: tuple[float, float] = MISSING  # about the peak acceleration, m/s^2
    yaw_rate: tuple[float, float] = MISSING  # peak heading rate, either way round, rad/s

    # walls, from the walk's start
    reach: float = 4.0  # sideways along x and y, before the walk is turned, m
    vertical: float = 0.6  # above or below the start height, m
    floor: float = 0.4  # lowest height of the walk, m

    # timing
    rest: float = 1.0  # time at rest at each end of the episode, s

    def draw(self, start: LoadState, episode: float) -> SplinePath:
        """Draw one random walk from the ranges.

        Args:
            start: The payload's state at the reset, whose height sets how low the walk may go.
            episode: Length of the episode, in s, which the walk fills from rest to rest.

        Returns:
            The walk as a spline path, to be placed at the payload by the command.
        """
        size, acceleration, yaw_rate = (
            np.random.uniform(*bounds) for bounds in (self.size, self.acceleration, self.yaw_rate)
        )
        interval, step = WALK_INTERVAL * np.sqrt(size / acceleration), WALK_STEP * size
        # The walk gets as many control points as fit before it must be at rest again.
        n = max(int((episode - 2 * self.rest) / interval) - SPLINE_DEGREE + 2, 2)

        # The walls sit around the walk's start, the origin here, but the floor depends on how
        # high the payload actually starts.
        lo = np.array([-self.reach, -self.reach, max(-self.vertical, self.floor - start.p[2])])
        hi = np.array([self.reach, self.reach, self.vertical])
        # The course turns less in height than sideways, since the height band is narrow.
        level = np.array([1.0, 1.0, 0.3])
        course = np.random.normal(size=3) * level
        course /= np.linalg.norm(course)
        points = [np.zeros(3)]
        for _ in range(n - 1):
            course += np.random.normal(size=3) * level
            course /= np.linalg.norm(course)
            p = points[-1] + step * np.random.uniform(0.5, 1.0) * course
            # A step that would cross a wall turns the course back along that axis.
            course[(p < lo) | (p > hi)] *= -1
            points.append(np.clip(p, lo, hi))

        rate, yaws = 0.0, [0.0]
        for _ in range(n - 1):
            rate = np.clip(rate + 0.3 * yaw_rate * np.random.normal(), -yaw_rate, yaw_rate)
            yaws.append(yaws[-1] + rate * interval)

        return SplinePath(
            points=np.array(points), yaws=np.array(yaws), interval=interval, start=self.rest
        )
