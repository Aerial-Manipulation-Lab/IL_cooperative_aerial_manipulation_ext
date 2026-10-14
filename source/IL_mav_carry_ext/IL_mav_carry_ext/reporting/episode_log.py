# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def episode_numbers(log: dict) -> dict[str, float]:
    """One reset's `extras["log"]` under the scripts' names.

    `Metrics/<command>/<name>` becomes `<name>`; `Episode_Termination/<term>` becomes
    `ended_by_<term>_share`, the share of envs whose last episode ended so. Terms that ended no
    episode are left out.
    """
    numbers = {}
    for key, value in log.items():
        kind, _, name = key.partition("/")
        if kind == "Metrics":
            numbers[name.partition("/")[2]] = float(value)
        elif kind == "Episode_Termination" and value:
            numbers[f"ended_by_{name}_share"] = float(value)
    return numbers


class EpisodeLog:
    """The per-episode numbers Isaac Lab's managers hand out at each reset, averaged over the resets
    since the last `means()`."""

    def __init__(self):
        self.entries = []

    def add(self, extras: dict, ended: torch.Tensor):
        """Call after every step; `extras["log"]` is only rebuilt on a step where an episode ended."""
        if bool(ended.any()):
            self.entries.append(episode_numbers(extras["log"]))

    def means(self) -> dict[str, float]:
        keys = dict.fromkeys(key for entry in self.entries for key in entry)
        means = {key: sum(entry.get(key, 0.0) for entry in self.entries) / len(self.entries) for key in keys}
        self.entries.clear()
        return means
