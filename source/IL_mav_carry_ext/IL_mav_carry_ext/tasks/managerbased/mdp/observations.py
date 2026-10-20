from collections.abc import Callable

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply
from python_mpc_cusadi import TuningCfg
from python_mpc_cusadi.ocp.horizon import Horizon

from ....plants import FLYCRANE_SIM

"""
Observations for the payload
"""

# Body indices found in the scene
# for the case when the rod is used
payload_idx = [1]
drone_idx = [20, 27, 34]


def payload_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_com_state_w.torch[:, payload_idx, 3:7].squeeze(1)
    payload_rot_matrix = matrix_from_quat(payload_quat)
    return payload_rot_matrix.view(env.num_envs, -1)


def payload_linear_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, payload_idx, 7:10].view(env.num_envs, -1)


def payload_angular_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, payload_idx, 10:].view(env.num_envs, -1)


"""
Observations for the drones: one row per drone, (num_envs, num_drones, k)
"""


def drone_orientations(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_quat = robot.data.body_com_state_w.torch[:, drone_idx, 3:7]
    drone_rot_matrix = matrix_from_quat(drone_quat)
    return drone_rot_matrix.flatten(-2)


def drone_linear_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 7:10]


def drone_angular_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 10:]


# relative drone positions


def payload_drone_rpos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the payload from the drone."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    payload_pos_world_frame = robot.data.body_com_state_w.torch[:, payload_idx, :3]
    return drone_pos_world_frame - payload_pos_world_frame


def drone_cable_dir(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Unit vector from each cable's attach point on the payload to its drone, world frame.

    The attach points are the MPC teacher's (`FLYCRANE_SIM`), so this is the cable
    geometry exactly as the MPC builds it from the measured endpoints under
    its taut-cable assumption.
    """
    return torch.nn.functional.normalize(-attach_point_rel_drone(env, asset_cfg), dim=-1)


def _attach_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Each drone's cable attach point on the payload as the sim has it, payload body frame: (num_drones, 3). For
    measurements; model-based terms read the teacher's plant instead."""
    return torch.tensor(np.stack([d.attach_point for d in FLYCRANE_SIM.drones]), dtype=torch.float32, device=env.device)


def attach_point_rel_drone(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Each drone's attach point on the payload relative to the drone, world frame: (num_envs, num_drones, 3).

    The cable as a vector, l s_i; on hardware a measured or estimated cable direction times the cable length.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, payload_idx[0]]
    attach_body = _attach_body(env)
    payload_quat = payload[:, None, 3:7].expand(-1, attach_body.shape[0], -1)
    attach_world = payload[:, None, :3] + quat_apply(payload_quat, attach_body.expand(env.num_envs, -1, -1))
    return attach_world - robot.data.body_com_state_w.torch[:, drone_idx, :3]


class attach_point_rel_drone_rate(ManagerTermBase):
    """The rate of `attach_point_rel_drone`, differenced over one step: how fast the attach point moves relative to the
    drone, (num_envs, num_drones, 3). Zero on an episode's first step."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last = torch.zeros(env.num_envs, len(drone_idx), 3, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        now = attach_point_rel_drone(env, asset_cfg)
        self.last[self.fresh] = now[self.fresh]
        self.fresh[:] = False
        rate = (now - self.last) / env.step_dt
        self.last = now
        return rate


# Observations of the reference the MPC tracks

REF_HORIZON_OFFSETS = Horizon.for_tuning(TuningCfg()).shooting_nodes
"""Times ahead of now at which the reference is observed: the MPC's own shooting
nodes, so the student sees exactly the stretch of reference each solve aims at."""


def payload_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The payload reference over the MPC's horizon, relative to the payload now.

    Per shooting node: position offset from the payload (3, env frame),
    reference velocity (3) and acceleration (3), attitude relative to the
    payload as rot6d (6), and reference body rate (3). Shape (num_envs,
    num_nodes * 18), node-major.

    Samples come from `LoadTrajectory.window`, the same lookup the MPC's
    reference sampler uses, at the same clock (`common_step_counter * step_dt`)
    the teacher solves at. rot6d is taken from the rotation matrix, which is
    the same for q and -q, so the MPC's hemisphere choice needs no mirroring.
    """
    term = env.command_manager.get_term(command_name)
    now = float(env.common_step_counter) * env.step_dt
    refs = [term.get_reference(i) for i in range(env.num_envs)]
    if any(ref is None for ref in refs):
        # only before the first reset, when the observation manager probes
        # this term for its shape: no ramp has been built yet
        return torch.zeros(env.num_envs, len(REF_HORIZON_OFFSETS) * 18, device=env.device)
    windows = [ref.window(now, REF_HORIZON_OFFSETS) for ref in refs]

    def stack(name):
        return torch.as_tensor(np.stack([getattr(w, name) for w in windows]), dtype=torch.float32, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, payload_idx[0]]
    payload_pos_env = payload[:, :3] - env.scene.env_origins

    # R_payload^T R_ref: the attitude still to go, in the payload's own frame
    rot_rel = matrix_from_quat(payload[:, None, 3:7]).transpose(-1, -2) @ matrix_from_quat(stack("q"))
    rot6d = rot_rel[..., :2].transpose(-1, -2).flatten(-2)  # first two columns, one after the other

    ref = torch.cat([stack("p") - payload_pos_env[:, None], stack("v"), stack("a"), rot6d, stack("w")], dim=-1)
    return ref.view(env.num_envs, -1)


def attach_point_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone's attach point should be over the MPC's horizon, from the payload reference and the attach
    point alone, so no payload state enters: per node its position relative to the drone, velocity and
    acceleration, world axes. (num_envs, num_drones, num_nodes * 9).

    c = p + R rho, c' = v + R (w x rho), c'' = a + R (alpha x rho + w x (w x rho)), with the reference's body-frame
    rates w and alpha.
    """
    term = env.command_manager.get_term(command_name)
    now = float(env.common_step_counter) * env.step_dt
    refs = [term.get_reference(i) for i in range(env.num_envs)]
    nodes, drones = len(REF_HORIZON_OFFSETS), len(drone_idx)
    if any(ref is None for ref in refs):
        # only before the first reset, when the observation manager probes this term for its shape
        return torch.zeros(env.num_envs, drones, nodes * 9, device=env.device)
    windows = [ref.window(now, REF_HORIZON_OFFSETS) for ref in refs]

    def stack(name):  # (num_envs, num_nodes, 1, k), broadcast over the drones
        return torch.as_tensor(np.stack([getattr(w, name) for w in windows]), dtype=torch.float32, device=env.device)[
            :, :, None
        ]

    rot = matrix_from_quat(stack("q"))
    # the attach points as the teacher's model has them: this is a reference, not a measurement
    model = torch.tensor(
        np.stack([d.attach_point for d in term.teacher.plant.drones]), dtype=torch.float32, device=env.device
    )
    rho = model.expand(env.num_envs, nodes, -1, -1)
    w, alpha = stack("w").expand_as(rho), stack("alpha").expand_as(rho)
    w_rho = torch.cross(w, rho, dim=-1)
    turn = lambda x: (rot @ x[..., None]).squeeze(-1)  # noqa: E731
    robot: Articulation = env.scene[asset_cfg.name]
    drone = robot.data.body_com_state_w.torch[:, drone_idx, :3] - env.scene.env_origins[:, None]
    c = stack("p") + turn(rho) - drone[:, None]
    c_dot = stack("v") + turn(w_rho)
    c_ddot = stack("a") + turn(torch.cross(alpha, rho, dim=-1) + torch.cross(w, w_rho, dim=-1))
    return torch.cat([c, c_dot, c_ddot], dim=-1).transpose(1, 2).flatten(2)


def drone_ref_window(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Each drone's reference (`MpcCommand.drone_references`) at the MPC's nodes from now: per drone and node p, v,
    a and the cable tension, env frame. (num_envs, num_drones, num_nodes, 10); zeros before the first reset."""
    term = env.command_manager.get_term(command_name)
    if any(ref is None for ref in term.drone_references):
        return torch.zeros(env.num_envs, len(drone_idx), len(REF_HORIZON_OFFSETS), 10, device=env.device)
    now = float(env.common_step_counter) * env.step_dt
    rows = [
        drones[ref.indices_at(now + REF_HORIZON_OFFSETS)]
        for ref, drones in zip(term.references, term.drone_references, strict=True)
    ]
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=env.device).transpose(1, 2)


def drone_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone should be over the horizon, relative to the drone now: per node position and velocity minus
    the drone's, and acceleration. Node 0 is the drone's own error. (num_envs, num_drones, num_nodes * 9)."""
    ref = drone_ref_window(env, command_name)[..., :9].clone()
    robot: Articulation = env.scene[asset_cfg.name]
    drones = robot.data.body_com_state_w.torch[:, drone_idx]
    ref[..., 0:3] -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    ref[..., 3:6] -= drones[:, :, None, 7:10]
    return ref.flatten(2)


def cable_tension_ref_horizon(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """How hard each drone's cable should pull over the horizon, N: (num_envs, num_drones, num_nodes)."""
    return drone_ref_window(env, command_name)[..., 9]


class attach_point_error(ManagerTermBase):
    """Each drone's attach point against its reference now: position and velocity of the reference minus the actual,
    the actual velocity from the drone's velocity and the cable vector differenced over one step.
    (num_envs, num_drones, 6)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rate = attach_point_rel_drone_rate(cfg, env)

    def reset(self, env_ids=None):
        self.rate.reset(env_ids)

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        ref = attach_point_ref_horizon(env, command_name, asset_cfg)[..., :6]  # node 0: position rel drone, velocity
        robot: Articulation = env.scene[asset_cfg.name]
        velocity = robot.data.body_com_state_w.torch[:, drone_idx, 7:10] + self.rate(env, asset_cfg)
        return torch.cat([ref[..., :3] - attach_point_rel_drone(env, asset_cfg), ref[..., 3:6] - velocity], dim=-1)


# The uniform student's input and label, one row per drone


def copy_to_each_drone(
    env: ManagerBasedRLEnv, term: Callable[..., torch.Tensor], params: dict | None = None
) -> torch.Tensor:
    """A term that is the same for every drone (the payload, its reference), (num_envs, k), copied
    to each: (num_envs, num_drones, k). `params` go to `term`."""
    return term(env, **(params or {}))[:, None].expand(-1, FLYCRANE_SIM.num_drones, -1)


def drone_attach_point(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Each drone's cable attach point on the payload, payload body frame, as the teacher's model has it:
    (num_envs, num_drones, 3). The one input that tells the drones apart."""
    points = np.stack([d.attach_point for d in env.command_manager.get_term(command_name).teacher.plant.drones])
    return torch.tensor(points, dtype=torch.float32, device=env.device).expand(env.num_envs, -1, -1)


def teacher_plan_rel_drone(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Each drone's teacher plan in the drone's own frame: the student's label and the teacher's
    action, (num_envs, num_drones, num_nodes * 12).

    Per node p, v, a, w as `MpcTeacher.plan`, with the drone's position and velocity now taken
    off p and v; the action term adds them back. From the true state, so noise on the policy
    group never reaches it. The command term solves before the obs are computed, so this is the
    plan for the state the policy group holds.
    """
    teacher = env.command_manager.get_term(command_name).teacher
    drones = teacher.robot.data.body_com_state_w.torch[:, teacher.falcon_idx]
    label = teacher.plan.clone()
    label[..., 0:3] -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    label[..., 3:6] -= drones[:, :, None, 7:10]
    return label.flatten(-2)
