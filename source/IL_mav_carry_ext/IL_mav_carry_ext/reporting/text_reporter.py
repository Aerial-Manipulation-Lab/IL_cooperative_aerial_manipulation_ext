# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


UNITS = {
    "_mps2": "m/s2",
    "_mps": "m/s",
    "_radps": "rad/s",
    "_hz": "Hz",
    "_deg": "deg",
    "_cm": "cm",
    "_min": "min",
    "_m": "m",
    "_s": "s",
}
"""Metric-name endings and the units they print as."""


class TextReporter:
    """One script's console output: status lines `[tag] ...` and metric lines `[tag label] name value unit | ...`.

    Every metric is formatted here, by one rule, so no script formats one itself: the unit comes off the
    end of its name and goes after the value, underscores print as spaces, a `_share` prints as a
    percentage, other floats with 3 significant digits (no decimals from 100 up), integers and flags as
    they are.
    """

    def __init__(self, tag: str):
        self.tag = tag

    def info(self, message: str):
        print(f"[{self.tag}] {message}", flush=True)

    def metrics(self, label: str, metrics: dict):
        line = " | ".join(format_metric(name, value) for name, value in metrics.items())
        print(f"[{self.tag} {label}] {line}", flush=True)


def format_metric(name: str, value) -> str:
    """One metric as `TextReporter` prints it: `name value unit`."""
    text = format_value(name, value)
    for ending, unit in UNITS.items():
        if name.endswith(ending):
            name, text = name[: -len(ending)], f"{text} {unit}"
            break
    return f"{name.replace('_', ' ')} {text}"


def format_value(name: str, value) -> str:
    """`value` as `TextReporter` prints it."""
    if isinstance(value, bool | int):
        return str(value)
    if name.endswith("_share"):
        return f"{value:.0%}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3g}"
