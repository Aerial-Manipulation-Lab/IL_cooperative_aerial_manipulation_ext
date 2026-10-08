# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The behaviour-cloning students: an MLP and a physics-informed TCN + FCN.

Both take and give physical units, unnormalised, as the thesis trains them."""

import torch
from torch import nn

from .features import LABEL_FRAME, PLAN_DIM, FeatureSpec


class BCPolicy(nn.Module):
    """Per-drone inputs -> that drone's plan label."""

    def __init__(self, input_dim: int, label_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.cfg = {"input_dim": input_dim, "label_dim": label_dim, "hidden": list(hidden)}
        layers, width = [], input_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ELU()]
            width = h
        layers.append(nn.Linear(width, label_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inputs (..., input_dim) -> labels (..., label_dim)."""
        return self.net(inputs)


class PINPolicy(nn.Module):
    """Per-drone inputs -> that drone's plan label, physics-informed; same interface as `BCPolicy`.

    After `PINN_FE` in the Imitation_learning repo: a TCN over the reference
    horizon and an FCN over the rest of the input make a latent, and an MLP of
    (latent, t) gives the drone's position offset and body rate t seconds
    ahead. Velocity and acceleration are that position's first and second time
    derivatives, so every plan is kinematically consistent by construction.
    """

    def __init__(
        self,
        input_dim: int,
        label_dim: int,
        ref_cols: list[int],
        vel_cols: list[int],
        node_times: list[float],
        hidden: tuple[int, ...] = (256, 256, 256),
        channels: int = 64,
        latent: int = 64,
    ):
        super().__init__()
        self.cfg = {
            "input_dim": input_dim,
            "label_dim": label_dim,
            "ref_cols": list(ref_cols),
            "vel_cols": list(vel_cols),
            "node_times": list(node_times),
            "hidden": list(hidden),
            "channels": channels,
            "latent": latent,
        }
        self.ref, self.vel = slice(*ref_cols), slice(*vel_cols)
        self.num_nodes = len(node_times)
        self.register_buffer("node_times", torch.tensor(node_times, dtype=torch.float32), persistent=False)
        ref_dim = ref_cols[1] - ref_cols[0]

        # TCN over the reference horizon, its nodes as the time axis
        self.tcn = nn.Sequential(
            nn.Conv1d(ref_dim // self.num_nodes, channels, 3, padding=1),
            nn.ELU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(channels * self.num_nodes, latent),
            nn.ELU(),
        )
        # FCN over the rest: the state now and the attach point
        self.fcn = nn.Sequential(nn.Linear(input_dim - ref_dim, 128), nn.ELU(), nn.Linear(128, latent), nn.ELU())
        # (latent, t) -> p (3), w (3); Tanh, as a = d2p/dt2 needs a smooth net
        # (SiLU is smooth too, but has no nested forward-mode AD in torch)
        layers, width = [], 2 * latent + 1
        for h in hidden:
            layers += [nn.Linear(width, h), nn.Tanh()]
            width = h
        layers.append(nn.Linear(width, 6))
        self.head = nn.Sequential(*layers)

    @classmethod
    def from_spec(cls, spec: FeatureSpec, hidden: tuple[int, ...] = (256, 256, 256)) -> "PINPolicy":
        ref, vel = spec.input_slice("payload_ref_horizon"), spec.input_slice("drone_linear_velocities")
        cols = [ref.start, ref.stop], [vel.start, vel.stop]
        return cls(spec.input_dim, spec.label_dim, *cols, spec.meta["node_offsets_s"], hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(-1, inputs.shape[-1])
        n, k = len(x), self.num_nodes
        ref = x[:, self.ref].reshape(n, k, -1).transpose(1, 2)  # (n, per-node dim, nodes)
        rest = torch.cat([x[:, : self.ref.start], x[:, self.ref.stop :]], dim=-1)
        z = torch.cat([self.tcn(ref), self.fcn(rest)], dim=-1).repeat_interleave(k, 0)
        v_now = x[:, self.vel].repeat_interleave(k, 0)

        # one row per (sample, node), and rows never mix: forward-mode AD with a
        # tangent of ones on t gives every row's d/dt in one pass, and nested, d2/dt2
        t = self.node_times.repeat(n)[:, None]
        ones = torch.ones_like(t)

        def head_and_rate(t):
            return torch.func.jvp(lambda t: self.head(torch.cat([z, t], dim=-1)), (t,), (ones,))

        (out, d_out), (_, dd_out) = torch.func.jvp(head_and_rate, (t,), (ones,))
        p, w, v, a = out[:, :3], out[:, 3:], d_out[:, :3], dd_out[:, :3]

        # the label frame of `FeatureSpec.plan_to_label`: v relative to the drone's velocity now
        label = torch.cat([p, v - v_now, a, w], dim=-1)
        return label.reshape(*inputs.shape[:-1], k * PLAN_DIM)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inputs (..., input_dim) -> labels (..., label_dim).

        Steps out of inference mode, which `play.py` flies under: forward-mode
        AD is off there, so v and a would silently come out as zero
        derivatives. The clone makes the inputs usable outside it.
        """
        with torch.inference_mode(False), torch.no_grad():
            return self(inputs.clone())


def build_model(spec: FeatureSpec, name: str) -> BCPolicy | PINPolicy:
    """An untrained student, "pin" or "mlp"; the one place train.py and dagger.py get it from."""
    if name == "mlp":
        return BCPolicy(spec.input_dim, spec.label_dim, hidden=(256, 256, 256))
    return PINPolicy.from_spec(spec, hidden=(256, 256, 256))


def save_checkpoint(path, model: BCPolicy | PINPolicy, spec: FeatureSpec, **extra):
    """Everything needed to fly the student without the training script."""
    torch.save(
        {
            "model": type(model).__name__,
            "model_cfg": model.cfg,
            "state_dict": model.state_dict(),
            "meta": spec.meta,
            "label_frame": LABEL_FRAME,
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[BCPolicy | PINPolicy, FeatureSpec, dict]:
    """(model, feature spec, the whole checkpoint dict) from `save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    frame = ckpt.get("label_frame", "payload")
    if frame != LABEL_FRAME:
        raise ValueError(
            f"{path} was trained with {frame}-relative labels, this code decodes "
            f"{LABEL_FRAME}-relative ones; retrain it"
        )
    cfg = ckpt["model_cfg"]
    if ckpt.get("model") == "PINPolicy":
        model = PINPolicy(**cfg).to(device)
    else:
        model = BCPolicy(cfg["input_dim"], cfg["label_dim"], tuple(cfg["hidden"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, FeatureSpec(ckpt["meta"]), ckpt
