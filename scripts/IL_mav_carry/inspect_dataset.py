# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity-check an NMPC demonstration dataset written by `scripts/record.py`.

Plain h5py and numpy, no simulator, so it runs in seconds. Reports what the
dataset holds and fails (exit code 1) if its parts disagree:

  - episodes: count, success rate, lengths
  - solver health: fraction of steps whose solve succeeded
  - shapes against the `.meta.json` next to the dataset
  - consistency: the recorded action is node 1 of the recorded plan, and
    matches the executed action wherever the teacher flew
  - label statistics per component, and no NaNs
  - how close to the goal the payload gets over an episode, from the obs

Example, from the repo root (the container has h5py, the host may not):

    ./docker/dev.sh run --rm isaac bash -c '$ISAAC_PY \
        IL_cooperative_aerial_manipulation_ext/scripts/IL_mav_carry/inspect_dataset.py \
        datasets/mpc_demos --plot 0'
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

LABEL_PARTS = {"p": slice(0, 3), "v": slice(3, 6), "a": slice(6, 9), "w": slice(9, 12)}
"""Per-node layout of `teacher_horizon`, as `dataset_metadata` records it."""


def obs_slices(meta):
    """{obs term name: column slice} from the recorded term layout."""
    slices, start = {}, 0
    for term in meta["obs_terms"]:
        slices[term["name"]] = slice(start, start + term["dim"])
        start += term["dim"]
    return slices, start


class Checks:
    """Collects pass/fail lines so every check runs before the verdict."""

    def __init__(self):
        self.failed = []

    def __call__(self, ok, what):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
        if not ok:
            self.failed.append(what)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check an NMPC demonstration dataset.")
    parser.add_argument("dataset", help="Dataset path, with or without .hdf5.")
    parser.add_argument("--plot", type=int, default=None, metavar="N", help="Save a plot of episode N as a PNG.")
    args = parser.parse_args()

    stem = os.path.splitext(args.dataset)[0]
    with open(stem + ".meta.json") as f:
        meta = json.load(f)
    slices, obs_dim = obs_slices(meta)
    D, K = meta["num_drones"], meta["num_nodes"]
    dt = meta["step_dt"]
    check = Checks()

    with h5py.File(stem + ".hdf5", "r") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[1]))
        if not demos:
            raise SystemExit(f"[ERROR]: no episodes in {stem}.hdf5")

        lengths, successes, ok_steps = [], [], []
        shapes_ok = consistent = executed_matches = True
        has_nan = False
        label_parts = {name: [] for name in LABEL_PARTS}
        # goal error per step index, over successful episodes only
        goal_err = []

        for name in demos:
            ep = f["data"][name]
            n = int(ep.attrs["num_samples"])
            lengths.append(n)
            success = bool(ep.attrs.get("success", False))
            successes.append(success)

            obs = ep["obs"][()]
            horizon = ep["teacher_horizon"][()]
            teacher_action = ep["teacher_action"][()]
            actions = ep["actions"][()]
            mpc_ok = ep["mpc_ok"][()].astype(bool)
            executed = ep["teacher_executed"][()].astype(bool)
            ok_steps.append(mpc_ok)

            shapes_ok &= obs.shape == (n, obs_dim)
            shapes_ok &= horizon.shape == (n, D, K, 12)
            shapes_ok &= teacher_action.shape == actions.shape == (n, D * 12)
            shapes_ok &= mpc_ok.shape == executed.shape == (n,)
            has_nan |= bool(np.isnan(obs).any() or np.isnan(horizon).any())

            # the action is node 1 of the plan: p, v, a per drone, then zeros
            per_drone = teacher_action.reshape(n, D, 12)
            consistent &= np.allclose(per_drone[..., :9], horizon[:, :, 1, :9], atol=1e-5)
            consistent &= np.all(per_drone[..., 9:] == 0)
            executed_matches &= np.allclose(actions[executed], teacher_action[executed], atol=1e-5)

            for part, cols in LABEL_PARTS.items():
                label_parts[part].append(horizon[mpc_ok][..., cols].reshape(-1, 3))
            if success and "payload_positional_error" in slices:
                goal_err.append(np.linalg.norm(obs[:, slices["payload_positional_error"]], axis=1))

    lengths = np.asarray(lengths)
    successes = np.asarray(successes)
    ok_frac = np.concatenate(ok_steps).mean()
    full_len = int(round(meta["episode_length_s"] / dt))

    print(f"\n{stem}.hdf5")
    print(f"  episodes {len(demos)}, successful {successes.sum()} ({successes.mean():.0%}), "
          f"steps {lengths.sum()} ({lengths.sum() * dt / 60:.1f} min of flight)")
    print(f"  episode length: min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()} "
          f"(full episode is {full_len} steps)")
    print(f"  steps with a successful solve: {ok_frac:.2%}")
    worst = np.argsort([s.mean() for s in ok_steps])[:3]
    print(f"  worst episodes by solve rate: " + ", ".join(f"{demos[i]} {ok_steps[i].mean():.1%}" for i in worst))

    print("\nchecks")
    check(shapes_ok, f"shapes match meta: obs (n, {obs_dim}), horizon (n, {D}, {K}, 12), action (n, {D * 12})")
    check(not has_nan, "no NaNs in obs or labels")
    check(consistent, "teacher_action is node 1 of teacher_horizon (p, v, a) with zeros after")
    check(executed_matches, "actions == teacher_action wherever the teacher flew")
    check(all(lengths[successes] == full_len), f"successful episodes all run the full {full_len} steps")

    print("\nlabel statistics (all drones and nodes, successful solves)")
    for part, rows in label_parts.items():
        rows = np.concatenate(rows)
        print(f"  {part}: mean {np.round(rows.mean(0), 3)}  std {np.round(rows.std(0), 3)}  "
              f"min {np.round(rows.min(0), 2)}  max {np.round(rows.max(0), 2)}")

    if goal_err:
        # successful episodes should all run the full length (checked above);
        # cut to the shortest so the table still prints if they do not
        common = min(len(e) for e in goal_err)
        goal_err = np.stack([e[:common] for e in goal_err])
        print(f"\ngoal error over the episode, successful episodes (ramp ends at 3 s)")
        for t in np.arange(0.0, common * dt, 1.0):
            k = min(int(round(t / dt)), common - 1)
            e = goal_err[:, k] * 100
            print(f"  t={t:3.0f}s  mean {e.mean():6.2f} cm  p90 {np.percentile(e, 90):6.2f} cm  max {e.max():6.2f} cm")
        final = goal_err[:, -1] * 100
        print(f"  end    mean {final.mean():6.2f} cm  p90 {np.percentile(final, 90):6.2f} cm  max {final.max():6.2f} cm")

    if args.plot is not None:
        plot_episode(stem, meta, slices, args.plot)

    if check.failed:
        print(f"\n{len(check.failed)} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


def plot_episode(stem, meta, slices, index):
    """Payload path against its goal and reference, and drone 0's setpoint, for one episode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(stem + ".hdf5", "r") as f:
        ep = f["data"][f"demo_{index}"]
        obs = ep["obs"][()]
        horizon = ep["teacher_horizon"][()]
    t = np.arange(len(obs)) * meta["step_dt"]
    payload = obs[:, slices["payload_position"]]
    goal = payload + obs[:, slices["payload_positional_error"]]
    # node 0 of the reference window, relative to the payload in the obs
    ref = payload + obs[:, slices["payload_ref_horizon"]].reshape(len(obs), -1, 18)[:, 0, :3]
    drone0 = horizon[:, 0, 1, :3]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    for i, (ax, axis) in enumerate(zip(axes, "xyz")):
        ax.plot(t, payload[:, i], label="payload")
        ax.plot(t, ref[:, i], "--", label="reference (node 0)")
        ax.plot(t, goal[:, i], ":", label="goal")
        ax.plot(t, drone0[:, i], alpha=0.6, label="drone 0 setpoint (node 1)")
        ax.set_ylabel(f"{axis} [m]")
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{os.path.basename(stem)} demo_{index}")
    path = f"{stem}_demo_{index}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
