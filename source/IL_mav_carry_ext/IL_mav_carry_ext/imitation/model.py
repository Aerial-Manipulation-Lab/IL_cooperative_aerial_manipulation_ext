# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The thesis's student (section 4.3, its `pinn_model.PINN_FE`), with a PINN or a plain MLP decoder.

Four TCN encoders, one per input stream of `FeatureSpec`, each to a 64-wide latent; the decoder maps
(latent, t) to the drone's plan at the node time t. The PINN decoder gives position and body rate, and
velocity and acceleration as the position's time derivatives; the MLP decoder (the thesis's baseline in
6.3) gives all four directly. Both take and give physical units, unnormalised, with position, velocity
and acceleration scaled by POSITION_SCALE at the output, as in the thesis.
"""

import itertools

import torch
from pytorch_tcn import TCN
from torch import nn

from .features import LOAD_DIM_WITH_VELOCITY, PLAN_DIM, FeatureSpec

POSITION_SCALE = 3.0
"""The thesis's output scale on position, velocity and acceleration."""


class ScaledTanh(nn.Module):
    """tanh(x) + alpha x tanh(x) with one trainable alpha per layer, the thesis's (after Stan)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return torch.tanh(x) + self.alpha * x * torch.tanh(x)


class ThesisPolicy(nn.Module):
    """Per-drone inputs -> that drone's plan label."""

    def __init__(
        self,
        streams: list,
        node_times: list[float],
        decoder: str,
        label_frame: str,
        channels: int = 256,
        latent: int = 64,
    ):
        super().__init__()
        self.cfg = {
            "streams": [list(s) for s in streams],
            "node_times": list(node_times),
            "decoder": decoder,
            "label_frame": label_frame,
            "channels": channels,
            "latent": latent,
        }
        self.streams, self.decoder, self.label_frame = streams, decoder, label_frame
        self.num_nodes = len(node_times)
        self.register_buffer("node_times", torch.tensor(node_times, dtype=torch.float32), persistent=False)
        # pytorch-tcn's defaults otherwise, as the thesis used it: weight norm, dropout 0.1, causal, ReLU
        self.encoders = nn.ModuleList(
            nn.Sequential(
                TCN(width, [channels, channels], 4 if name == "reference" else 8, input_shape="NLC"),
                nn.Flatten(),
                nn.Linear(channels * length, latent),
                nn.ReLU(),
            )
            for name, length, width in streams
        )
        width_in = len(streams) * latent + 1
        if decoder == "pin":
            layers, widths = [], [width_in, 256, 256, 128, 128, 128]
            for a, b in itertools.pairwise(widths):
                layers += [nn.Linear(a, b), ScaledTanh()]
            self.head = nn.Sequential(*layers, nn.Linear(128, 6))  # p, w
        else:
            layers, width = [], width_in
            for _ in range(5):
                layers += [nn.Linear(width, 128), nn.ReLU()]
                width = 128
            self.head = nn.Sequential(*layers, nn.Linear(128, PLAN_DIM))  # p, v, a, w

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(-1, inputs.shape[-1])
        n, k = len(x), self.num_nodes
        parts = x.split([length * width for _, length, width in self.streams], dim=-1)
        latents = [
            encoder(part.reshape(n, length, width))
            for encoder, part, (_, length, width) in zip(self.encoders, parts, self.streams, strict=True)
        ]
        z = torch.cat(latents, dim=-1).repeat_interleave(k, 0)
        t = self.node_times.repeat(n)[:, None]

        if self.decoder == "pin":
            # one row per (sample, node), and rows never mix: forward-mode AD with a tangent of ones on t
            # gives every row's d/dt in one pass, and nested, d2/dt2
            ones = torch.ones_like(t)

            def head_and_rate(t):
                return torch.func.jvp(lambda t: self.head(torch.cat([z, t], dim=-1)), (t,), (ones,))

            (out, d_out), (_, dd_out) = torch.func.jvp(head_and_rate, (t,), (ones,))
            p_v_a = torch.cat([out[:, :3], d_out[:, :3], dd_out[:, :3]], -1)
            label = torch.cat([POSITION_SCALE * p_v_a, out[:, 3:]], -1)
            if self.label_frame == "drone":
                # its label's velocity is relative to the drone's now: the newest sample of the drone stream
                _, length, width = self.streams[1]
                v_now = parts[1].reshape(n, length, width)[:, -1, 4:7]
                label = torch.cat([label[:, :3], label[:, 3:6] - v_now.repeat_interleave(k, 0), label[:, 6:]], -1)
        else:
            out = self.head(torch.cat([z, t], dim=-1))
            label = torch.cat([POSITION_SCALE * out[:, :9], out[:, 9:]], -1)
        return label.reshape(*inputs.shape[:-1], k * PLAN_DIM)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inputs (..., input_dim) -> labels (..., label_dim).

        Steps out of inference mode, which `play.py` flies under: forward-mode AD is off there, so the
        PINN's v and a would silently come out as zero derivatives. The clone makes the inputs usable
        outside it.
        """
        with torch.inference_mode(False), torch.no_grad():
            return self(inputs.clone())


def build_model(spec: FeatureSpec, name: str) -> ThesisPolicy:
    """An untrained student with a "pin" or "mlp" decoder; the one place train.py and dagger.py get it from."""
    return ThesisPolicy(spec.streams, spec.meta["node_offsets_s"], name, spec.label_frame)


def save_checkpoint(path, model: ThesisPolicy, spec: FeatureSpec, **extra):
    """Everything needed to fly the student without the training script."""
    torch.save(
        {
            "model": type(model).__name__,
            "model_cfg": model.cfg,
            "state_dict": model.state_dict(),
            "meta": spec.meta,
            "label_frame": spec.label_frame,
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[ThesisPolicy, FeatureSpec, dict]:
    """(model, feature spec, the whole checkpoint dict) from `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ThesisPolicy(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # whether it was given the payload's velocities shows in its load stream's width
    load_width = {name: width for name, _, width in ckpt["model_cfg"]["streams"]}["load"]
    spec = FeatureSpec(ckpt["meta"], ckpt["label_frame"], payload_velocity=load_width == LOAD_DIM_WITH_VELOCITY)
    return model, spec, ckpt
