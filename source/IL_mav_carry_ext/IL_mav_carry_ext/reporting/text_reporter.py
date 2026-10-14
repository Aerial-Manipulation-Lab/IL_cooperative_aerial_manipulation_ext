# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class TextReporter:
    """One script's console output: status lines `[tag] ...` and metric lines `[tag label] name value | ...`.

    Every number is formatted here, by one rule, so no script formats one itself: the unit is in its
    name, a `_share` prints as a percentage, other floats with 3 significant digits (no decimals from
    100 up), integers and flags as they are.
    """

    def __init__(self, tag: str):
        self.tag = tag

    def info(self, message: str):
        print(f"[{self.tag}] {message}", flush=True)

    def metrics(self, label: str, metrics: dict):
        line = " | ".join(f"{name} {format_value(name, value)}" for name, value in metrics.items())
        print(f"[{self.tag} {label}] {line}", flush=True)


def format_value(name: str, value) -> str:
    """`value` as `TextReporter` prints it."""
    if isinstance(value, bool | int):
        return str(value)
    if name.endswith("_share"):
        return f"{value:.0%}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3g}"
