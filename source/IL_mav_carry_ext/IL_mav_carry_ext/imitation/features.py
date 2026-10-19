# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The student's rows by name: which columns of a drone's policy row are which term, and which node and
quantity each column of its label is."""

from dataclasses import dataclass

import torch

PLAN_DIM = 12
"""Per plan node: p(3), v(3), a(3), w(3)."""

PLAN_PARTS = {"p": slice(0, 3), "v": slice(3, 6), "a": slice(6, 9), "w": slice(9, 12)}
"""Where a node's position, velocity, acceleration and body rate sit in it."""


@dataclass(frozen=True)
class FeatureSpec:
    """One drone's row into and out of the student.

    Built from the env, so it is what the env produces, and saved with every dataset and checkpoint, so a
    checkpoint only flies an env whose spec equals its own. It names columns and nothing else: the
    observation manager builds the rows, and the checkpoint next to it names the model.
    """

    policy_terms: tuple[tuple[str, int], ...]
    """The policy group's terms in column order, each with its width per drone."""
    num_drones: int
    node_offsets_s: tuple[float, ...]
    """The times ahead of now of the label plan's nodes."""
    step_dt: float
    """The control step."""

    @classmethod
    def from_env(cls, env) -> "FeatureSpec":
        """The spec of `env`'s policy and label groups."""
        manager = env.observation_manager
        teacher = env.command_manager.get_term("pose_command").teacher
        terms = zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"], strict=True)
        return cls(
            policy_terms=tuple((name, int(dims[-1])) for name, dims in terms),
            num_drones=teacher.num_drones,
            node_offsets_s=tuple(float(t) for t in teacher.node_offsets_s),
            step_dt=float(env.step_dt),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        """The spec `to_dict` wrote."""
        return cls(
            policy_terms=tuple((term["name"], term["dim"]) for term in d["policy_terms"]),
            num_drones=d["num_drones"],
            node_offsets_s=tuple(d["node_offsets_s"]),
            step_dt=d["step_dt"],
        )

    def to_dict(self) -> dict:
        """As JSON-ready builtins, for a dataset's meta.json and a checkpoint."""
        return {
            "policy_terms": [{"name": name, "dim": dim} for name, dim in self.policy_terms],
            "num_drones": self.num_drones,
            "node_offsets_s": list(self.node_offsets_s),
            "step_dt": self.step_dt,
        }

    @property
    def input_dim(self) -> int:
        return sum(dim for _, dim in self.policy_terms)

    @property
    def num_nodes(self) -> int:
        return len(self.node_offsets_s)

    @property
    def label_dim(self) -> int:
        return self.num_nodes * PLAN_DIM

    def slice(self, term: str) -> slice:
        """Where `term` sits in a policy row: `x[..., spec.slice(term)]`."""
        start = 0
        for name, dim in self.policy_terms:
            if name == term:
                return slice(start, start + dim)
            start += dim
        raise KeyError(f"{term!r} is not a policy term; they are {[name for name, _ in self.policy_terms]}")

    def plan(self, label: torch.Tensor) -> torch.Tensor:
        """Label rows (..., label_dim) as plans (..., num_nodes, PLAN_DIM): `spec.plan(y)[..., 1, PLAN_PARTS["v"]]`
        is node 1's velocity."""
        return label.unflatten(-1, (self.num_nodes, PLAN_DIM))
