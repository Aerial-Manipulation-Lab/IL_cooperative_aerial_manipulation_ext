# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The behaviour-cloning student: an MLP with its normalisation built in.

The normalisation statistics are buffers of the module, so a saved checkpoint
takes raw per-drone inputs and returns labels in physical units -- nothing
outside it has to remember how the training data was scaled.
"""

import torch
from torch import nn

from .features import LABEL_FRAME, FeatureSpec


class BCPolicy(nn.Module):
    """Per-drone inputs -> that drone's plan label, both unnormalised.

    `forward` works on normalised tensors (what the loss sees); `predict` wraps
    it in the stored normalisation for everything else.
    """

    def __init__(self, input_dim: int, label_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.cfg = {"input_dim": input_dim, "label_dim": label_dim, "hidden": list(hidden)}
        layers, width = [], input_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ELU()]
            width = h
        layers.append(nn.Linear(width, label_dim))
        self.net = nn.Sequential(*layers)
        # identity until `set_normalisation`, so a fresh model is still usable
        self.register_buffer("in_mean", torch.zeros(input_dim))
        self.register_buffer("in_std", torch.ones(input_dim))
        self.register_buffer("out_mean", torch.zeros(label_dim))
        self.register_buffer("out_std", torch.ones(label_dim))

    def set_normalisation(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Per-dimension mean and std from the training set.

        Dimensions that never vary (a std near zero) are left unscaled rather
        than blown up: they carry nothing to learn from.
        """
        def stats(x):
            std = x.std(0)
            return x.mean(0), torch.where(std > 1e-6, std, torch.ones_like(std))

        self.in_mean, self.in_std = stats(inputs)
        self.out_mean, self.out_std = stats(labels)

    def normalise_inputs(self, inputs):
        return (inputs - self.in_mean) / self.in_std

    def normalise_labels(self, labels):
        return (labels - self.out_mean) / self.out_std

    def forward(self, inputs_normalised: torch.Tensor) -> torch.Tensor:
        return self.net(inputs_normalised)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Raw inputs (..., input_dim) -> labels (..., label_dim) in physical units."""
        return self.net(self.normalise_inputs(inputs)) * self.out_std + self.out_mean


def save_checkpoint(path, model: BCPolicy, spec: FeatureSpec, **extra):
    """Everything needed to fly the student without the training script."""
    torch.save({"model_cfg": model.cfg, "state_dict": model.state_dict(), "meta": spec.meta,
                "label_frame": LABEL_FRAME, **extra}, path)


def load_checkpoint(path, device="cpu") -> tuple[BCPolicy, FeatureSpec, dict]:
    """(model, feature spec, the whole checkpoint dict) from `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # checkpoints from before the frame was recorded were all trained payload-relative
    frame = ckpt.get("label_frame", "payload")
    if frame != LABEL_FRAME:
        raise ValueError(f"{path} was trained with {frame}-relative labels, this code decodes "
                         f"{LABEL_FRAME}-relative ones; retrain it")
    cfg = ckpt["model_cfg"]
    model = BCPolicy(cfg["input_dim"], cfg["label_dim"], tuple(cfg["hidden"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, FeatureSpec(ckpt["meta"]), ckpt
