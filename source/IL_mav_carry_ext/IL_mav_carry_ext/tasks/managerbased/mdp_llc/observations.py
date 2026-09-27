import numpy as np
import torch

from python_mpc_cusadi import TuningCfg
from python_mpc_cusadi.ocp.horizon import Horizon

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_conjugate, quat_inv, quat_mul

from ....plants import FLYCRANE_SIM
from .utils import get_drone_pdist, get_drone_rpos

"""
Observations for the payload
"""

# Body indices found in the scene
# payload_idx = [1]
# drone_idx = [71, 72, 73]
# base_rope_idx = [8, 9, 10]

# for the case when the rod is used
payload_idx = [1]
drone_idx = [20, 27, 34]
base_rope_idx = [8, 9, 10]


def payload_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload pose xyz, quat in env frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_world_frame = robot.data.body_com_state_w.torch[:, payload_idx, :3].squeeze(1)
    payload_env_frame = payload_world_frame - env.scene.env_origins
    return payload_env_frame.view(env.num_envs, -1)


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


def payload_linear_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w.torch[:, payload_idx, 0:3].view(env.num_envs, -1)


def payload_angular_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w.torch[:, payload_idx, 3:].view(env.num_envs, -1)


def payload_positional_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload position error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_com_state_w.torch[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_pos = env.command_manager.get_command(command_name)[..., :3]
    positional_error = desired_pos - payload_pos_env
    return positional_error


def payload_orientation_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload orientation error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_com_state_w.torch[:, payload_idx, 3:7].squeeze(1)
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    payload_rot_matrix = matrix_from_quat(payload_quat)
    desired_rot_matrix = matrix_from_quat(desired_quat)
    orientation_error = torch.matmul(desired_rot_matrix, payload_rot_matrix.transpose(1, 2)).view(env.num_envs, -1)

    return orientation_error


def payload_velocity_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload velocity error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_vel_world = robot.data.body_com_state_w.torch[:, payload_idx, 7:10].squeeze(1)
    desired_vel = env.command_manager.get_command(command_name)[..., 7:10]
    vel_error = desired_vel - payload_vel_world
    return vel_error


def payload_linear_velocity_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_com_state_w.torch[:, payload_idx, 7:10].squeeze(1)
    desired_lin_vel = env.command_manager.get_command(command_name)[..., 7:10]
    lin_vel_error = desired_lin_vel - payload_lin_vel

    return lin_vel_error


def payload_angular_velocity_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_com_state_w.torch[:, payload_idx, 10:13].squeeze(1)
    desired_ang_vel = env.command_manager.get_command(command_name)[..., 10:13]
    ang_vel_error = desired_ang_vel - payload_ang_vel

    return ang_vel_error


def cable_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angle of cable between cable and payload."""
    robot: Articulation = env.scene[asset_cfg.name]
    rope_orientations_world = robot.data.body_com_state_w.torch[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_com_state_w.torch[:, payload_idx, 3:7].repeat(1, 3, 1).view(-1, 4)
    payload_orientation_inv = quat_inv(payload_orientation_world)
    rope_orientations_payload = quat_mul(
        payload_orientation_inv, rope_orientations_world
    )  # cable angles relative to payload
    return rope_orientations_payload.view(env.num_envs, -1)


"""
Observations for the drones
"""


def drone_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone positions xyz in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    drone_pos_env_frame = drone_pos_world_frame - env.scene.env_origins.unsqueeze(1)
    return drone_pos_env_frame.view(env.num_envs, -1)


def drone_orientations(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_quat = robot.data.body_com_state_w.torch[:, drone_idx, 3:7]
    drone_rot_matrix = matrix_from_quat(drone_quat)
    return drone_rot_matrix.view(env.num_envs, -1)


def drone_linear_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 7:10].view(env.num_envs, -1)


def drone_angular_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 10:].view(env.num_envs, -1)


def drone_linear_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w.torch[:, drone_idx, 0:3].view(env.num_envs, -1)


def drone_angular_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w.torch[:, drone_idx, 3:].view(env.num_envs, -1)


# relative drone positions


def payload_drone_rpos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the payload from the drone."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    payload_pos_world_frame = robot.data.body_com_state_w.torch[:, payload_idx, :3]
    rpos = drone_pos_world_frame - payload_pos_world_frame
    return rpos.view(env.num_envs, -1)


def drone_rpos_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the drones from each other."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    return rpos.view(env.num_envs, -1)


def drone_pdist_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Euclidean distance between drones."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = torch.norm(rpos, dim=-1, keepdim=True)
    return pdist.view(env.num_envs, -1)


def drone_cable_dir(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Unit vector from each cable's attach point on the payload to its drone, world frame.

    The attach points are the MPC teacher's (`FLYCRANE_SIM`), so this is the cable
    geometry exactly as the MPC builds it from the measured endpoints under
    its taut-cable assumption. Shape (num_envs, 3 * num_drones), drone-major.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, payload_idx[0]]
    attach_body = torch.tensor(np.stack([d.attach_point for d in FLYCRANE_SIM.drones]),
                               dtype=torch.float32, device=env.device)
    num_drones = attach_body.shape[0]
    payload_quat = payload[:, None, 3:7].expand(-1, num_drones, -1)
    attach_world = payload[:, None, :3] + quat_apply(payload_quat, attach_body.expand(env.num_envs, -1, -1))
    cable = robot.data.body_com_state_w.torch[:, drone_idx, :3] - attach_world
    return torch.nn.functional.normalize(cable, dim=-1).view(env.num_envs, -1)


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

    ref = torch.cat(
        [stack("p") - payload_pos_env[:, None], stack("v"), stack("a"), rot6d, stack("w")], dim=-1
    )
    return ref.view(env.num_envs, -1)


# Observations for when sampling multiple points on a trajectory


def payload_positional_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload position error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_com_state_w.torch[:, payload_idx, :3]
    payload_pos_env = payload_pos_world - env.scene.env_origins.unsqueeze(1)
    desired_pos = env.command_manager.get_command(command_name)[..., :3]
    positional_error = (desired_pos - payload_pos_env).view(env.num_envs, -1)
    return positional_error


def payload_orientation_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload orientation error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    payload_quat = robot.data.body_com_state_w.torch[:, payload_idx, 3:7].repeat(1, desired_quat.shape[1], 1)
    orientation_error = quat_mul(desired_quat.view(-1, 4), quat_conjugate(payload_quat.view(-1, 4))).view(
        env.num_envs, -1
    )
    return orientation_error


def payload_linear_velocity_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_com_state_w.torch[:, payload_idx, 7:10]
    desired_lin_vel = env.command_manager.get_command(command_name)[..., 7:10]
    lin_vel_error = (desired_lin_vel - payload_lin_vel).view(env.num_envs, -1)
    return lin_vel_error


def payload_angular_velocity_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_com_state_w.torch[:, payload_idx, 10:13]
    desired_ang_vel = env.command_manager.get_command(command_name)[..., 10:13]
    ang_vel_error = (desired_ang_vel - payload_ang_vel).view(env.num_envs, -1)
    return ang_vel_error


def payload_linear_acc_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear acceleration error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_acc = robot.data.body_acc_w.torch[:, payload_idx, 0:3]
    desired_lin_acc = env.command_manager.get_command(command_name)[..., 13:16]
    lin_acc_error = (desired_lin_acc - payload_lin_acc).view(env.num_envs, -1)
    return lin_acc_error


def payload_angular_acc_error_traj(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular acceleration error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_acc = robot.data.body_acc_w.torch[:, payload_idx, 3:6]
    desired_ang_acc = env.command_manager.get_command(command_name)[..., 3:6]
    ang_acc_error = (desired_ang_acc - payload_ang_acc).view(env.num_envs, -1)
    return ang_acc_error


def obstacle_rpos(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg = SceneEntityCfg("wall")) -> torch.Tensor:
    """Get the relative distance to the obstacle"""
    obstacle = env.scene[obstacle_cfg.name]
    robot: Articulation = env.scene["robot"]
    payload_pos_env = robot.data.body_com_state_w.torch[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    obstacle_pos = obstacle.data.body_com_state_w.torch[:, 0, :3] - env.scene.env_origins
    rpos = obstacle_pos - payload_pos_env
    return rpos.view(env.num_envs, -1)


def obstacle_rpos_2(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg = SceneEntityCfg("wall_2")) -> torch.Tensor:
    """Get the relative distance to the obstacle"""
    obstacle = env.scene[obstacle_cfg.name]
    robot: Articulation = env.scene["robot"]
    payload_pos_env = robot.data.body_com_state_w.torch[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    obstacle_pos = obstacle.data.body_com_state_w.torch[:, 0, :3] - env.scene.env_origins
    rpos = obstacle_pos - payload_pos_env
    return rpos.view(env.num_envs, -1)


def obstacle_geometry(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg = SceneEntityCfg("wall")) -> torch.Tensor:
    """Get the obstacle size parameters"""
    wall_dimensions = torch.tensor([0.1, 10.0, 1.5], device=env.device).repeat(env.num_envs, 1)
    return wall_dimensions


# policy terms


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the previous action taken by the policy."""
    return env.action_manager._action
