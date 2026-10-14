# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from torch.utils.tensorboard import SummaryWriter


class TensorBoardReporter:
    """A run's metrics as TensorBoard scalars, under the names the console shows them by."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def metrics(self, metrics: dict, step: int):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def close(self):
        self.writer.close()
