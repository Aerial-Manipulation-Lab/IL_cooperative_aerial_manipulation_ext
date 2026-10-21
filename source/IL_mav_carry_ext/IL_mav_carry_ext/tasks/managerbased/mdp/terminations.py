import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_inv, quat_mul

from ....assets import CABLE_IDX, DRONE_IDX, PAYLOAD_IDX

"""
Ground.
"""


def drone_too_low(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when any drone comes closer to the ground than the threshold.

    Args:
        threshold: Lowest allowed height of a drone, in m above the ground.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    origins = env.scene.env_origins.unsqueeze(1)
    drone_pos = robot.data.body_com_state_w.torch[:, DRONE_IDX, :3] - origins
    return (drone_pos[..., 2] < threshold).any(dim=1)


def payload_too_low(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when the payload comes closer to the ground than the threshold.

    Args:
        threshold: Lowest allowed height of the payload, in m above the ground.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    payload_pos = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX[0], :3] - env.scene.env_origins
    return payload_pos[:, 2] < threshold


"""
Cables.
"""


def cable_angle_at_drone_too_large(
    env: ManagerBasedRLEnv,
    threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when any cable leans too far away from its drone's vertical axis.

    The cable's roll and pitch are measured in its drone's frame, and the episode ends when the
    cosine of either one drops below the threshold.

    Args:
        threshold: Cosine of the largest allowed angle, so 0 allows up to 90 degrees.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    cable_quat = robot.data.body_com_state_w.torch[:, CABLE_IDX, 3:7].reshape(-1, 4)
    drone_quat = robot.data.body_com_state_w.torch[:, DRONE_IDX, 3:7].reshape(-1, 4)
    # The cable's attitude in its drone's frame, of which only roll and pitch matter.
    roll, pitch, _ = euler_xyz_from_quat(quat_mul(quat_inv(drone_quat), cable_quat))
    too_large = (torch.cos(roll) < threshold) | (torch.cos(pitch) < threshold)
    return too_large.view(env.num_envs, -1).any(dim=1)


def cable_angle_at_payload_too_large(
    env: ManagerBasedRLEnv,
    threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when any cable leans too far away from the payload's vertical axis.

    The cable's roll and pitch are measured in the payload's frame, and the episode ends when the
    cosine of either one drops below the threshold.

    Args:
        threshold: Cosine of the largest allowed angle, so 0 allows up to 90 degrees.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    cable_quat = robot.data.body_com_state_w.torch[:, CABLE_IDX, 3:7].reshape(-1, 4)
    payload_quat = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX, 3:7]
    payload_quat = payload_quat.expand(-1, len(CABLE_IDX), -1).reshape(-1, 4)
    # The cable's attitude in the payload's frame, of which only roll and pitch matter.
    roll, pitch, _ = euler_xyz_from_quat(quat_mul(quat_inv(payload_quat), cable_quat))
    too_large = (torch.cos(roll) < threshold) | (torch.cos(pitch) < threshold)
    return too_large.view(env.num_envs, -1).any(dim=1)


def cables_too_close(
    env: ManagerBasedRLEnv,
    threshold: float = 0.0,
    num_points: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when two cables come closer to each other than the threshold.

    Each cable is sampled at evenly spaced points from its lower end to its drone, and the
    episode ends when any two points on different cables are closer than the threshold.

    Args:
        threshold: Smallest allowed distance between two cables, in m.
        num_points: Points sampled along each cable.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    origins = env.scene.env_origins.unsqueeze(1)
    bottom = robot.data.body_com_state_w.torch[:, CABLE_IDX, :3] - origins
    top = robot.data.body_com_state_w.torch[:, DRONE_IDX, :3] - origins
    along = torch.linspace(0, 1, num_points, device=env.device).view(1, 1, num_points, 1)
    points = bottom.unsqueeze(2) + along * (top - bottom).unsqueeze(2)
    distances = torch.cdist(points.flatten(1, 2), points.flatten(1, 2))
    # Points on the same cable are always close to each other, so those pairs are left out.
    cable_of_point = torch.arange(len(CABLE_IDX), device=env.device).repeat_interleave(num_points)
    same_cable = cable_of_point.unsqueeze(0) == cable_of_point.unsqueeze(1)
    distances = distances.masked_fill(same_cable, float("inf"))
    return distances.flatten(1).min(dim=1).values < threshold


"""
Drones and payload.
"""


def drones_too_close(
    env: ManagerBasedRLEnv,
    threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when two drones come closer to each other than the threshold.

    Args:
        threshold: Smallest allowed distance between two drones, in m.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    drone_pos = robot.data.body_com_state_w.torch[:, DRONE_IDX, :3]
    distances = torch.cdist(drone_pos, drone_pos)
    # A drone's distance to itself is always zero, so those pairs are left out.
    itself = torch.eye(len(DRONE_IDX), dtype=torch.bool, device=env.device)
    distances = distances.masked_fill(itself, float("inf"))
    return distances.flatten(1).min(dim=1).values < threshold


def payload_out_of_bounds(
    env: ManagerBasedRLEnv,
    threshold: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode when the payload leaves a box around its env's origin.

    Args:
        threshold: Half the box's side, in m along each world axis.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    payload_pos = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX[0], :3] - env.scene.env_origins
    return (payload_pos.abs() > threshold).any(dim=-1)


"""
Teacher.
"""


def teacher_solve_failed(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """End the episode when the teacher's last solve failed, since its plan cannot be trusted.

    Returns:
        Whether each env ends, of shape (num_envs,).
    """
    return ~env.command_manager.get_term(command_name).teacher.ok
