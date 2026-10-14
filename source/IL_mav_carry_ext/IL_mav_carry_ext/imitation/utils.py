# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training pieces shared by scripts/train.py and scripts/dagger.py."""

import json
import os

import h5py
import torch


def bc_loss(model, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE between the student's prediction and the teacher's label, in physical units as in the thesis."""
    return torch.nn.functional.mse_loss(model(inputs), labels)


def load_episodes(paths: list[str]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """(policy rows, label rows) per episode over all files, as (steps, num_drones, dim) CPU tensors, and the metadata.

    Every step whose solve succeeded is kept, also from episodes that ended early: under DAgger those
    are where the student crashed. Episodes without steps (reset again right after a reset) are skipped.
    """
    meta, episodes = None, []
    for path in paths:
        stem = os.path.splitext(path)[0]
        with open(stem + ".meta.json") as f:
            file_meta = json.load(f)
        if meta is None:
            meta = file_meta
        elif file_meta["policy_terms"] != meta["policy_terms"]:
            raise ValueError(f"{stem}: policy terms differ from {paths[0]}")
        with h5py.File(stem + ".hdf5", "r") as f:
            print(f"[INFO]: loading {len(f['data'])} episodes from {stem}.hdf5", flush=True)
            for n, ep in enumerate(f["data"].values(), start=1):
                if n % 50 == 0:
                    print(f"[INFO]:   {n}/{len(f['data'])} episodes", flush=True)
                if ep.attrs["num_samples"] == 0:
                    continue
                ok = ep["mpc_ok"][()].astype(bool)
                obs = torch.as_tensor(ep["obs"][()][ok], dtype=torch.float32)
                label = torch.as_tensor(ep["label"][()][ok], dtype=torch.float32)
                episodes.append((obs, label))
    print(f"[INFO]: {len(episodes)} episodes loaded", flush=True)
    return episodes, meta
