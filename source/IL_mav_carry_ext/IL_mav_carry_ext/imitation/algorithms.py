# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The learning algorithms the scripts drive, after RSL-RL's: each holds the student and its optimiser,
and `update()` does one round of learning and returns its numbers under the shared names. `Dagger` also
acts, as it learns while flying; `BehaviourCloning` learns from recordings only."""

import torch

from .utils import Buffer, evaluate, fit


class Dagger:
    """DAgger with a gate (Ross et al.): the student flies where its whole plan is within `gate` of the
    teacher's, the teacher elsewhere, and every state is labelled with the teacher's plan. Until the
    first update the teacher flies alone, which makes the first round behaviour cloning.
    """

    def __init__(self, model, lr: float, gate: float, keep_every: int, epochs: int, batch_size: int):
        self.model = model
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self.gate, self.keep_every, self.epochs, self.batch_size = gate, keep_every, epochs, batch_size
        self.buffer = Buffer()
        self.updates = 0
        self.steps = self.decisions = self.flown = self.fresh = 0  # this round's, and where its examples start

    def act(self, obs: dict[str, torch.Tensor], ok: torch.Tensor) -> torch.Tensor:
        """This step's action. Every `keep_every`-th step, the examples whose teacher solve succeeded
        (`ok`, per env) go into the buffer."""
        x, label = obs["policy"], obs["label"]
        if self.steps % self.keep_every == 0:
            self.buffer.add(x[ok], label[ok])
        self.steps += 1
        self.decisions += len(ok)
        if self.updates == 0:
            return label.flatten(1)  # the teacher flies its own plan
        pred = self.model.predict(x)
        student_flies = ((pred - label).pow(2).mean((1, 2)) < self.gate) & ok
        self.flown += int(student_flies.sum())
        return torch.where(student_flies[:, None, None], pred, label).flatten(1)

    def update(self) -> dict[str, float]:
        """The student on the states flown since the last update, before it trains on them; then
        `epochs` passes over every example so far."""
        inputs, labels = self.buffer.tensors()
        new = inputs[self.fresh :], labels[self.fresh :]
        unseen = evaluate(self.model, *new) if len(new[0]) else {}
        train_loss = fit(self.model, self.optimiser, inputs, labels, self.epochs, self.batch_size)
        metrics = {
            "student_share": self.flown / self.decisions if self.decisions else 0.0,
            **unseen,
            "train_loss": train_loss,
            "buffer": len(inputs),
        }
        self.updates += 1
        self.steps = self.decisions = self.flown = 0
        self.fresh = len(inputs)
        return metrics


class BehaviourCloning:
    """Behaviour cloning on recorded teacher flights: each update is one pass over the training examples,
    then an evaluation on the held-out ones; the learning rate halves when that stops improving."""

    def __init__(self, model, train_data, val_data, lr: float, batch_size: int):
        self.model = model
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=0.5, patience=5)
        self.train_data, self.val_data, self.batch_size = train_data, val_data, batch_size

    def update(self) -> dict[str, float]:
        train_loss = fit(self.model, self.optimiser, *self.train_data, 1, self.batch_size)
        metrics = {
            "train_loss": train_loss,
            **evaluate(self.model, *self.val_data, self.batch_size),
            "lr": self.optimiser.param_groups[0]["lr"],
        }
        self.scheduler.step(metrics["val_loss"])
        return metrics
