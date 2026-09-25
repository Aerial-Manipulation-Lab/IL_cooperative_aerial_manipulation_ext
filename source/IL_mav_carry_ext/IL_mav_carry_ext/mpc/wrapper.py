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
    """Runs one NMPC per env in place of a policy.

    `step()` takes no action: the teacher's solve is the action. Envs whose
    solve fails are reset (a bad solve makes the rest of the episode worth-
    less), and every done env -- failed, terminated or truncated -- has its
    policy handed the ramped reference the command term built during its reset.

    Pairs with `RecordVideo` as base -> teacher -> recorder, so the recorder
    sees the standard five-element tuples.
    """

    def __init__(self, env, **teacher_kwargs):
        super().__init__(env)
        self.teacher = MpcTeacher(env, **teacher_kwargs)
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
        obs, rew, terminated, truncated, info = self.env.step(self.teacher.act(self.time))
        self.count += 1

        # the env auto-resets on termination; failed solves have to be reset
        # by us, and both cases then need the same re-seeding
        done = terminated | truncated
        failed = self.teacher.last_failed
        if failed:
            failed_ids = torch.tensor(failed, dtype=torch.long, device=self.teacher.robot.data.root_pos_w.torch.device)
            self.env.unwrapped.reset(env_ids=failed_ids)
            done[failed_ids] = True

        if bool(done.any()):
            ids = torch.nonzero(done).flatten().tolist()
            self.teacher.seed(ids)
            if self.teacher.verbose:
                cmd = self.unwrapped.command_manager.get_command(MpcTeacher.COMMAND_NAME)
                for i in ids:
                    reason = "solver failure" if i in failed else ", ".join(fired_terminations(self.env, i))
                    goal = cmd[i, :3].cpu().numpy().round(2)
                    print(
                        f"[INFO]: env {i} episode ended at t={self.time:.2f}s "
                        f"({reason}), new goal {goal}"
                    )

        info["mpc_pos_err"] = self.teacher.last_pos_err
        info["mpc_failed"] = list(failed)
        return obs, rew, terminated, truncated, info
