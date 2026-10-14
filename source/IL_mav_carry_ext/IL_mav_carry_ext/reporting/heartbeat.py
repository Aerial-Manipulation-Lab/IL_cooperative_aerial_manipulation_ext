# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time


class Heartbeat:
    """Due once every `seconds` of wall time: when a script's next progress line is."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last = time.time()

    def due(self) -> bool:
        now = time.time()
        if now - self.last < self.seconds:
            return False
        self.last = now
        return True
