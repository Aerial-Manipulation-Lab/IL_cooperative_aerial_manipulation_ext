# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium wrapper that closes the loop with the NMPC teacher."""

import gymnasium as gym
import torch

from .teacher import MpcTeacher


def fired_terminations(env, env_id):
    """Which termination terms ended env_id's episode. Diagnostic only, so it never raises."""
    try:
        manager = env.unwrapped.termination_manager
        return [name for name in manager.active_terms if bool(manager.get_term(name)[env_id])] or ["(none)"]
    except Exception as exc:  # the manager API is not worth crashing a run over
        return [f"(unavailable: {exc})"]


class MpcTeacherWrapper(gym.Wrapper):
    """Runs one NMPC per env, as the policy or as the label for one.

    Every step the teacher solves every env, and that solve is the label.
    Which action is *executed* is DAgger's mixing: `step(None)` flies the
    teacher everywhere; `step(student_action)` flies, per env and per step,
    the teacher with probability `beta` and the student otherwise. Behaviour
    cloning is `step(None)`; DAgger is a student action with `beta` decayed
    over rounds.

    Before stepping the env, the step's teacher output is left on the
    unwrapped env, where recorder terms read it inside `env.step`:

    - `teacher_action`: (num_envs, action_dim), the teacher's action for this
      step, in the action term's format.
    - `teacher_horizon`: (num_envs, num_drones, num_nodes, 12), the whole
      plan that action is node 1 of; see `MpcTeacher.last_horizon`.
    - `mpc_ok`: (num_envs,) bool, False where the solve failed.
    - `teacher_executed`: (num_envs,) bool, True where the teacher's action
      was the one applied.

    Envs whose solve fails are reset (a bad solve makes the rest of the
    episode worthless), and every done env -- failed, terminated or truncated
    -- has its policy handed the ramped reference the command term built
    during its reset.

    Pairs with `RecordVideo` as base -> teacher -> recorder, so the recorder
    sees the standard five-element tuples.
    """

    def __init__(self, env, beta: float = 1.0, reset_on_failed_solve: bool = True, **teacher_kwargs):
        super().__init__(env)
        self.teacher = MpcTeacher(env, **teacher_kwargs)
        self.beta = beta
        """Probability the teacher's action is the one executed, per env and step,
        when a student action is given. Set it between rounds to decay it."""
        self.reset_on_failed_solve = reset_on_failed_solve
        """Reset an env whose solve failed. Right while the teacher flies or its
        labels are recorded; off to evaluate a student, whose episode a failed
        teacher solve says nothing about (that step's label is still flagged
        in `mpc_ok`)."""
        self.count = 0

    @property
    def time(self) -> float:
        """The env's own clock: every actor timestamps off the step counter.

        Not an accumulator -- the command term stamps its ramps with the same
        expression, and the two must never disagree. `env.step` increments the
        counter before any reset runs, so a reference sampled during a reset
        starts exactly at the next solve's time.
        """
        return float(self.unwrapped.common_step_counter) * self.teacher.step_dt

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        self.count = 0
        # the command term built every env's ramp during the reset; the
        # policies just have not been told about them yet
        self.teacher.seed(range(self.teacher.num_envs))
        return ret

    def step(self, action=None):
        base = self.unwrapped
        teacher_action = self.teacher.act(self.time)
        failed = self.teacher.last_failed

        if action is None:
            executed = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
            applied = teacher_action
        else:
            executed = torch.rand(base.num_envs, device=base.device) < self.beta
            applied = torch.where(executed[:, None], teacher_action, action.to(base.device))

        mpc_ok = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        mpc_ok[failed] = False
        base.teacher_action = teacher_action
        # a copy: the teacher refills that buffer in place every solve, and
        # whoever keeps this step's plan must not see the next step's
        base.teacher_horizon = self.teacher.last_horizon.clone()
        base.mpc_ok = mpc_ok
        base.teacher_executed = executed

        obs, rew, terminated, truncated, info = self.env.step(applied)
        self.count += 1

        # the env auto-resets on termination; failed solves have to be reset
        # by us, and both cases then need the same re-seeding
        done = terminated | truncated
        # an env the step already ended was auto-reset into a fresh episode
        # the failed solve never acted on: resetting it again would only
        # export that barely begun episode, flagged by the step's own time-out
        failed_ids = torch.tensor(failed if self.reset_on_failed_solve else [], dtype=torch.long, device=base.device)
        failed_ids = failed_ids[~done[failed_ids]]
        if len(failed_ids) > 0:
            # the obs the step returned was computed before this reset; the
            # reset recomputes it, and the student must act on the new one
            obs, _ = base.reset(env_ids=failed_ids)
            done[failed_ids] = True

        if bool(done.any()):
            ids = torch.nonzero(done).flatten().tolist()
            self.teacher.seed(ids)
            if self.teacher.verbose:
                cmd = base.command_manager.get_command(MpcTeacher.COMMAND_NAME)
                reset_by_us = set(failed_ids.tolist())
                for i in ids:
                    reason = "solver failure" if i in reset_by_us else ", ".join(fired_terminations(self.env, i))
                    goal = cmd[i, :3].cpu().numpy().round(2)
                    print(
                        f"[INFO]: env {i} episode ended at t={self.time:.2f}s "
                        f"({reason}), new goal {goal}"
                    )

        info["mpc_pos_err"] = self.teacher.last_pos_err
        info["mpc_failed"] = list(failed)
        return obs, rew, terminated, truncated, info
