from collections.abc import Callable

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

from ....plants import FLYCRANE_SIM

# Body indices in the scene's articulation, with the rod cables

payload_idx = [1]
drone_idx = [20, 27, 34]


# Payload


def copy_to_each_drone(
    env: ManagerBasedRLEnv, term: Callable[..., torch.Tensor], params: dict | None = None
) -> torch.Tensor:
    """A term shared by all drones, copied to each drone."""
    return term(env, **(params or {}))[:, None].expand(-1, FLYCRANE_SIM.num_drones, -1)


def payload_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload rotation matrix, flattened, in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_com_state_w.torch[:, payload_idx, 3:7].squeeze(1)
    payload_rot_matrix = matrix_from_quat(payload_quat)
    return payload_rot_matrix.view(env.num_envs, -1)


def payload_linear_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, payload_idx, 7:10].view(env.num_envs, -1)


def payload_angular_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, payload_idx, 10:].view(env.num_envs, -1)


def payload_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload reference at the MPC's nodes: position, velocity, acceleration, rot6d attitude and body rate, relative to
    the payload."""
    nodes = _payload_references_from_command(env, command_name)
    payload = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, payload_idx[0]]
    position = nodes["p"] - (payload[:, :3] - env.scene.env_origins)[:, None]
    # the attitude still to go, in the payload's own frame
    payload_rotation = matrix_from_quat(payload[:, None, 3:7])
    attitude = _rot6d_from_rotation_matrix(payload_rotation.transpose(-1, -2) @ nodes["rotation"])
    return torch.cat([position, nodes["v"], nodes["a"], attitude, nodes["w"]], dim=-1).flatten(1)


# Drones: one row per drone, (num_envs, num_drones, k)


def drone_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone rotation matrices, flattened, in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_quat = robot.data.body_com_state_w.torch[:, drone_idx, 3:7]
    drone_rot_matrix = matrix_from_quat(drone_quat)
    return drone_rot_matrix.flatten(-2)


def drone_linear_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 7:10]


def drone_angular_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, drone_idx, 10:]


def drone_position_rel_payload(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone positions relative to the payload, in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, drone_idx, :3]
    payload_pos_world_frame = robot.data.body_com_state_w.torch[:, payload_idx, :3]
    return drone_pos_world_frame - payload_pos_world_frame


def drone_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone reference at the MPC's nodes: position and velocity relative to the drone, and acceleration."""
    nodes = _drone_references_from_command(env, command_name)
    drones = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, drone_idx]
    position = nodes["p"] - (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    velocity = nodes["v"] - drones[:, :, None, 7:10]
    return torch.cat([position, velocity, nodes["a"]], dim=-1).flatten(2)


# Cables


def cable_attach_point(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Cable attach points of the teacher's model, in payload frame."""
    points = np.stack([d.attach_point for d in env.command_manager.get_term(command_name).teacher.plant.drones])
    return torch.tensor(points, dtype=torch.float32, device=env.device).expand(env.num_envs, -1, -1)


def cable_direction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Unit vector from each cable's attach point on the payload to its drone (in world frame)."""
    return torch.nn.functional.normalize(-cable_attach_rel_drone(env, asset_cfg), dim=-1)


def cable_attach_rel_drone(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cable attach point relative to its drone, in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, payload_idx[0]]
    attach_body = _cable_attach_sim(env)
    payload_quat = payload[:, None, 3:7].expand(-1, attach_body.shape[0], -1)
    attach_world = payload[:, None, :3] + quat_apply(payload_quat, attach_body.expand(env.num_envs, -1, -1))
    return attach_world - robot.data.body_com_state_w.torch[:, drone_idx, :3]


class cable_attach_rel_drone_rate(ManagerTermBase):
    """Rate of `cable_attach_rel_drone` over the last step, zero on an episode's first step."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last = torch.zeros(env.num_envs, len(drone_idx), 3, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        now = cable_attach_rel_drone(env, asset_cfg)
        self.last[self.fresh] = now[self.fresh]
        self.fresh[:] = False
        rate = (now - self.last) / env.step_dt
        self.last = now
        return rate


def cable_attach_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Attach point reference at the MPC's nodes: position relative to the drone, velocity and acceleration, in world
    frame."""
    teacher = env.command_manager.get_term(command_name).teacher
    nodes = _payload_references_from_command(env, command_name)
    # broadcast the reference over the drones: (num_envs, num_nodes, num_drones, 3)
    p, v, a, rotation = (nodes[name][:, :, None] for name in ("p", "v", "a", "rotation"))
    # the attach points as the teacher's model has them: this is a reference, not a measurement
    rho = torch.tensor(np.stack([d.attach_point for d in teacher.plant.drones]), dtype=torch.float32, device=env.device)
    rho = rho.expand(env.num_envs, teacher.num_nodes, -1, -1)
    w, alpha = nodes["w"][:, :, None].expand_as(rho), nodes["alpha"][:, :, None].expand_as(rho)
    w_rho = torch.cross(w, rho, dim=-1)
    turn = lambda x: (rotation @ x[..., None]).squeeze(-1)  # noqa: E731
    drone = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, drone_idx, :3] - env.scene.env_origins[:, None]
    c = p + turn(rho) - drone[:, None]
    c_dot = v + turn(w_rho)
    c_ddot = a + turn(torch.cross(alpha, rho, dim=-1) + torch.cross(w, w_rho, dim=-1))
    return torch.cat([c, c_dot, c_ddot], dim=-1).transpose(1, 2).flatten(2)


class cable_attach_error(ManagerTermBase):
    """Attach point position and velocity error against its reference, in world frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rate = cable_attach_rel_drone_rate(cfg, env)

    def reset(self, env_ids=None):
        self.rate.reset(env_ids)

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        ref = cable_attach_ref_horizon(env, command_name, asset_cfg)[..., :6]  # node 0: position rel drone, velocity
        robot: Articulation = env.scene[asset_cfg.name]
        velocity = robot.data.body_com_state_w.torch[:, drone_idx, 7:10] + self.rate(env, asset_cfg)
        return torch.cat([ref[..., :3] - cable_attach_rel_drone(env, asset_cfg), ref[..., 3:6] - velocity], dim=-1)


def cable_tension_ref_horizon(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Cable tension reference at the MPC's nodes, in N."""
    return _drone_references_from_command(env, command_name)["tension"]


# Teacher: the student's label


def teacher_plan_rel_drone(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Teacher plan per drone, position and velocity relative to the drone."""
    teacher = env.command_manager.get_term(command_name).teacher
    drones = teacher.robot.data.body_com_state_w.torch[:, teacher.falcon_idx]
    label = teacher.plan.clone()
    label[..., 0:3] -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    label[..., 3:6] -= drones[:, :, None, 7:10]
    return label.flatten(-2)


# Helpers


def _payload_references_from_command(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    """Payload reference at the MPC's nodes by field, in env frame; zeros before the first reset."""
    term = env.command_manager.get_term(command_name)
    refs = [term.get_reference(i) for i in range(env.num_envs)]
    if any(ref is None for ref in refs):  # the observation manager probes the shapes before the first reset
        zeros = torch.zeros(env.num_envs, term.teacher.num_nodes, 3, device=env.device)
        nodes = dict.fromkeys(("p", "v", "a", "w", "alpha"), zeros)
        nodes["rotation"] = torch.zeros(env.num_envs, term.teacher.num_nodes, 3, 3, device=env.device)
        return nodes
    now = float(env.common_step_counter) * env.step_dt
    windows = [ref.window(now, term.teacher.node_offsets_s) for ref in refs]

    def stack(name):
        return torch.as_tensor(np.stack([getattr(w, name) for w in windows]), dtype=torch.float32, device=env.device)

    nodes = {name: stack(name) for name in ("p", "v", "a", "w", "alpha")}
    nodes["rotation"] = matrix_from_quat(stack("q"))
    return nodes


def _drone_references_from_command(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    """Drone reference at the MPC's nodes by field, in env frame; zeros before the first reset."""
    term = env.command_manager.get_term(command_name)
    if any(ref is None for ref in term.drone_references):  # the observation manager probes the shapes first
        rows = torch.zeros(env.num_envs, len(drone_idx), term.teacher.num_nodes, 10, device=env.device)
    else:
        now = float(env.common_step_counter) * env.step_dt
        rows = [
            drones[ref.indices_at(now + term.teacher.node_offsets_s)]
            for ref, drones in zip(term.references, term.drone_references, strict=True)
        ]
        rows = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=env.device).transpose(1, 2)
    return {"p": rows[..., 0:3], "v": rows[..., 3:6], "a": rows[..., 6:9], "tension": rows[..., 9]}


def _rot6d_from_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, one after the other."""
    return rotation[..., :2].transpose(-1, -2).flatten(-2)


def _cable_attach_sim(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cable attach points of the sim's payload, in payload frame."""
    return torch.tensor(np.stack([d.attach_point for d in FLYCRANE_SIM.drones]), dtype=torch.float32, device=env.device)
