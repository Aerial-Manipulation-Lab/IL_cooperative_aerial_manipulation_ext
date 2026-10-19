from collections.abc import Callable

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
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
    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, payload_idx[0]]
    attach_body = torch.tensor(
        np.stack([d.attach_point for d in FLYCRANE_SIM.drones]), dtype=torch.float32, device=env.device
    )
    num_drones = attach_body.shape[0]
    payload_quat = payload[:, None, 3:7].expand(-1, num_drones, -1)
    attach_world = payload[:, None, :3] + quat_apply(payload_quat, attach_body.expand(env.num_envs, -1, -1))
    cable = robot.data.body_com_state_w.torch[:, drone_idx, :3] - attach_world
    return torch.nn.functional.normalize(cable, dim=-1)


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


def drone_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """`payload_ref_horizon` with each node's position relative to the drone instead of the payload, so the
    payload's position enters nowhere: (num_envs, num_drones, num_nodes * 18)."""
    ref = payload_ref_horizon(env, command_name, asset_cfg).view(env.num_envs, 1, len(REF_HORIZON_OFFSETS), 18)
    ref = ref.repeat(1, len(drone_idx), 1, 1)
    # reference - drone = (reference - payload) - (drone - payload)
    ref[..., :3] -= payload_drone_rpos(env, asset_cfg)[:, :, None]
    return ref.flatten(2)


# The uniform student's input and label, one row per drone


def copy_to_each_drone(
    env: ManagerBasedRLEnv, term: Callable[..., torch.Tensor], params: dict | None = None
) -> torch.Tensor:
    """A term that is the same for every drone (the payload, its reference), (num_envs, k), copied
    to each: (num_envs, num_drones, k). `params` go to `term`."""
    return term(env, **(params or {}))[:, None].expand(-1, FLYCRANE_SIM.num_drones, -1)


def drone_attach_point(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Each drone's cable attach point on the payload, payload body frame, as the MPC has it
    (`FLYCRANE_SIM`): (num_envs, num_drones, 3). The one input that tells the drones apart."""
    points = np.stack([d.attach_point for d in FLYCRANE_SIM.drones])
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
