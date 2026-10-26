from collections.abc import Callable

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply
from python_mpc_cusadi import PlantCfg

from ....assets import DRONE_IDX, PAYLOAD_IDX
from ....plants import FLYCRANE_SIM

"""
Payload.
"""


def copy_shared_term_to_each_drone(
    env: ManagerBasedRLEnv, term: Callable[..., torch.Tensor], params: dict | None = None
) -> torch.Tensor:
    """Give each drone its own copy of a term that is the same for all drones.

    The student takes one row of inputs per drone. Terms about the payload or its reference are
    the same for every drone, so they are computed once and copied into each drone's row.

    Args:
        term: Observation term with one row per env, of shape (num_envs, k).
        params: Keyword arguments passed on to `term`.

    Returns:
        The term's row repeated for each drone, of shape (num_envs, num_drones, k).
    """
    return term(env, **(params or {}))[:, None].expand(-1, FLYCRANE_SIM.num_drones, -1)


def payload_orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How the payload is rotated, as the rotation matrix from its body frame to the world.

    Returns:
        Rotation matrix flattened row by row, of shape (num_envs, 9).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX, 3:7].squeeze(1)
    payload_rot_matrix = matrix_from_quat(payload_quat)
    return payload_rot_matrix.view(env.num_envs, -1)


def payload_linear_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How fast the payload is moving, in m/s, along the world axes.

    Returns:
        Linear velocity of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, PAYLOAD_IDX, 7:10].view(env.num_envs, -1)


def payload_angular_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How fast the payload is turning, in rad/s, about the world axes.

    Returns:
        Angular velocity of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, PAYLOAD_IDX, 10:].view(env.num_envs, -1)


def payload_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where the payload should be at each of the MPC's nodes, seen from where it is now.

    Each node holds, in this order:

    - the position still to go in m, along the world axes;
    - the reference velocity in m/s;
    - the reference acceleration in m/s^2;
    - the rotation still to go as rot6d, in the payload frame;
    - the reference body rate in rad/s.

    Returns:
        Reference of shape (num_envs, num_nodes * 18).
    """
    nodes = _get_payload_references(env, command_name)
    payload = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, PAYLOAD_IDX[0]]
    position = nodes["p"] - (payload[:, :3] - env.scene.env_origins)[:, None]
    # The attitude is taken relative to the payload, so the student sees what is still to turn.
    payload_rotation = matrix_from_quat(payload[:, None, 3:7])
    attitude_to_go = payload_rotation.transpose(-1, -2) @ nodes["rotation"]
    attitude = _convert_rotation_matrix_to_rot6d(attitude_to_go)
    return torch.cat([position, nodes["v"], nodes["a"], attitude, nodes["w"]], dim=-1).flatten(1)


"""
Drones, one row per drone.
"""


def drone_orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How each drone is rotated, as the rotation matrix from its body frame to the world.

    Returns:
        Rotation matrices flattened row by row, of shape (num_envs, num_drones, 9).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    drone_quat = robot.data.body_com_state_w.torch[:, DRONE_IDX, 3:7]
    drone_rot_matrix = matrix_from_quat(drone_quat)
    return drone_rot_matrix.flatten(-2)


def drone_linear_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How fast each drone is moving, in m/s, along the world axes.

    Returns:
        Linear velocities of shape (num_envs, num_drones, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, DRONE_IDX, 7:10]


def drone_angular_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How fast each drone is turning, in rad/s, about the world axes.

    Returns:
        Angular velocities of shape (num_envs, num_drones, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_com_state_w.torch[:, DRONE_IDX, 10:]


def drone_position_rel_payload(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone is, seen from the payload, in m along the world axes.

    Returns:
        Positions of shape (num_envs, num_drones, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_com_state_w.torch[:, DRONE_IDX, :3]
    payload_pos_world_frame = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX, :3]
    return drone_pos_world_frame - payload_pos_world_frame


def drone_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone should be at each of the MPC's nodes, seen from where it is now.

    Each node holds, in this order, along the world axes:

    - the position still to go in m;
    - the velocity still to gain in m/s;
    - the reference acceleration in m/s^2.

    Returns:
        Reference of shape (num_envs, num_drones, num_nodes * 9).
    """
    nodes = _get_drone_references(env, command_name)
    drones = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, DRONE_IDX]
    position = nodes["p"] - (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    velocity = nodes["v"] - drones[:, :, None, 7:10]
    return torch.cat([position, velocity, nodes["a"]], dim=-1).flatten(2)


"""
Cables.
"""


def cable_attach_point(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """Where each drone's cable is fixed on the payload, in m in the payload frame.

    These come from the teacher's model of the plant. They also tell the drones apart, since
    all drones share one student.

    Returns:
        Attach points of shape (num_envs, num_drones, 3).
    """
    plant = env.command_manager.get_term(command_name).teacher.plant
    return _get_attach_points(env, plant).expand(env.num_envs, -1, -1)


def cable_direction(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Which way each cable runs, from its attach point up to its drone, along the world axes.

    The direction is a unit vector.

    Returns:
        Directions of shape (num_envs, num_drones, 3).
    """
    return torch.nn.functional.normalize(-cable_attach_rel_drone(env, asset_cfg), dim=-1)


def cable_attach_rel_drone(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone's attach point on the payload is, seen from the drone, in m.

    This is the cable as a vector, its direction times its length, along the world axes.

    Returns:
        Attach points of shape (num_envs, num_drones, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    payload = robot.data.body_com_state_w.torch[:, PAYLOAD_IDX[0]]
    # This is a measurement, so it uses the sim's own geometry and not the teacher's model.
    attach_body = _get_attach_points(env, FLYCRANE_SIM)
    payload_quat = payload[:, None, 3:7].expand(-1, attach_body.shape[0], -1)
    attach_body = attach_body.expand(env.num_envs, -1, -1)
    attach_world = payload[:, None, :3] + quat_apply(payload_quat, attach_body)
    return attach_world - robot.data.body_com_state_w.torch[:, DRONE_IDX, :3]


class cable_attach_rel_drone_rate(ManagerTermBase):
    """How fast each attach point moves as seen from its drone, in m/s along the world axes.

    The rate is the change of `cable_attach_rel_drone` since the last step, divided by the step
    time. It is zero on an episode's first step, which has no last step.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Start with no last step for any env.

        Args:
            cfg: Configuration of the observation term.
            env: Environment the term observes.
        """
        super().__init__(cfg, env)
        self.last = torch.zeros(env.num_envs, len(DRONE_IDX), 3, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    """
    Operations.
    """

    def reset(self, env_ids=None):
        """Forget the last step of the envs being reset.

        Args:
            env_ids: Envs being reset, all if None.
        """
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Compute how fast each attach point moves as seen from its drone.

        Returns:
            Rates in m/s, of shape (num_envs, num_drones, 3).
        """
        now = cable_attach_rel_drone(env, asset_cfg)
        # A fresh env has no previous step, so its rate starts at zero.
        self.last[self.fresh] = now[self.fresh]
        self.fresh[:] = False
        rate = (now - self.last) / env.step_dt
        self.last = now
        return rate


def cable_attach_ref_horizon(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Where each drone's attach point should be at each of the MPC's nodes, seen from the drone.

    The attach point's reference follows from the payload's reference and where the cable is
    fixed on the payload, so no measurement of the payload enters. Each node holds, in this
    order, along the world axes:

    - the position of the reference attach point, seen from the drone now, in m;
    - its velocity in m/s;
    - its acceleration in m/s^2.

    Returns:
        Reference of shape (num_envs, num_drones, num_nodes * 9).
    """
    teacher = env.command_manager.get_term(command_name).teacher
    nodes = _get_payload_references(env, command_name)
    # The reference gets a drone axis, (num_envs, num_nodes, num_drones, 3), to broadcast over.
    p, v, a, rotation = (nodes[name][:, :, None] for name in ("p", "v", "a", "rotation"))
    # This is a reference, so it uses the teacher's model of the attach points.
    rho = _get_attach_points(env, teacher.plant).expand(env.num_envs, teacher.num_nodes, -1, -1)
    w, alpha = nodes["w"][:, :, None].expand_as(rho), nodes["alpha"][:, :, None].expand_as(rho)
    w_rho = torch.cross(w, rho, dim=-1)
    turn = lambda x: (rotation @ x[..., None]).squeeze(-1)  # noqa: E731
    drones = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, DRONE_IDX, :3]
    drone = drones - env.scene.env_origins[:, None]
    c = p + turn(rho) - drone[:, None]
    c_dot = v + turn(w_rho)
    c_ddot = a + turn(torch.cross(alpha, rho, dim=-1) + torch.cross(w, w_rho, dim=-1))
    return torch.cat([c, c_dot, c_ddot], dim=-1).transpose(1, 2).flatten(2)


class cable_attach_error(ManagerTermBase):
    """How far each attach point is from where it should be now, along the world axes.

    It holds, in this order:

    - the position error in m;
    - the velocity error in m/s.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Keep track of the attach point's rate, which the velocity error needs.

        Args:
            cfg: Configuration of the observation term.
            env: Environment the term observes.
        """
        super().__init__(cfg, env)
        self.rate = cable_attach_rel_drone_rate(cfg, env)

    """
    Operations.
    """

    def reset(self, env_ids=None):
        """Forget the last step of the envs being reset.

        Args:
            env_ids: Envs being reset, all if None.
        """
        self.rate.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute how far each attach point is from where it should be now.

        Returns:
            Position and velocity errors, of shape (num_envs, num_drones, 6).
        """
        # Node 0 of the reference is now, its position relative to the drone and its velocity.
        ref = cable_attach_ref_horizon(env, command_name, asset_cfg)[..., :6]
        robot: Articulation = env.scene[asset_cfg.name]
        velocity = robot.data.body_com_state_w.torch[:, DRONE_IDX, 7:10] + self.rate(env, asset_cfg)
        position_error = ref[..., :3] - cable_attach_rel_drone(env, asset_cfg)
        return torch.cat([position_error, ref[..., 3:6] - velocity], dim=-1)


def cable_tension_ref_horizon(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """How hard each cable should pull at each of the MPC's nodes, in N.

    Returns:
        Tensions of shape (num_envs, num_drones, num_nodes).
    """
    return _get_drone_references(env, command_name)["tension"]


"""
Teacher, the student's label.
"""


def teacher_plan_rel_drone(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """What the teacher plans for each drone at each of the MPC's nodes, seen from the drone now.

    This is the student's label. Each node holds, in this order, along the world axes:

    - the position still to go in m;
    - the velocity still to gain in m/s;
    - the planned acceleration in m/s^2;
    - the planned body rate in rad/s.

    Returns:
        Plan of shape (num_envs, num_drones, num_nodes * 12).
    """
    teacher = env.command_manager.get_term(command_name).teacher
    drones = teacher.robot.data.body_com_state_w.torch[:, teacher.falcon_idx]
    label = teacher.plan.clone()
    label[..., 0:3] -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    label[..., 3:6] -= drones[:, :, None, 7:10]
    return label.flatten(-2)


"""
Helpers.
"""


def _get_payload_references(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    """Get the payload's reference at each of the MPC's nodes, in the env frame.

    Args:
        env: Environment holding the command.
        command_name: Name of the MPC command holding the references.

    Returns:
        Fields `p`, `v`, `a`, `w` and `alpha` of shape (num_envs, num_nodes, 3) and `rotation` of
        shape (num_envs, num_nodes, 3, 3). Zeros before the first reset.
    """
    term = env.command_manager.get_term(command_name)
    refs = [term.get_reference(i) for i in range(env.num_envs)]
    # The observation manager probes the shapes before the first reset, when there are no
    # references yet.
    if any(ref is None for ref in refs):
        zeros = torch.zeros(env.num_envs, term.teacher.num_nodes, 3, device=env.device)
        nodes = dict.fromkeys(("p", "v", "a", "w", "alpha"), zeros)
        num_nodes = term.teacher.num_nodes
        nodes["rotation"] = torch.zeros(env.num_envs, num_nodes, 3, 3, device=env.device)
        return nodes
    now = float(env.common_step_counter) * env.step_dt
    windows = [ref.get_window(now, term.teacher.node_offsets_s) for ref in refs]

    def stack(name):
        values = np.stack([getattr(w, name) for w in windows])
        return torch.as_tensor(values, dtype=torch.float32, device=env.device)

    nodes = {name: stack(name) for name in ("p", "v", "a", "w", "alpha")}
    nodes["rotation"] = matrix_from_quat(stack("q"))
    return nodes


def _get_drone_references(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    """Get where each drone should be at each of the MPC's nodes, in the env frame.

    Args:
        env: Environment holding the command.
        command_name: Name of the MPC command holding the drone references.

    Returns:
        Fields `p`, `v` and `a` of shape (num_envs, num_drones, num_nodes, 3) and `tension` of
        shape (num_envs, num_drones, num_nodes). Zeros before the first reset.
    """
    term = env.command_manager.get_term(command_name)
    # The observation manager probes the shapes before the first reset, when there are no
    # references yet.
    if any(ref is None for ref in term.drone_references):
        num_nodes = term.teacher.num_nodes
        rows = torch.zeros(env.num_envs, len(DRONE_IDX), num_nodes, 10, device=env.device)
    else:
        now = float(env.common_step_counter) * env.step_dt
        rows = [
            drones[ref.get_indices_at(now + term.teacher.node_offsets_s)]
            for ref, drones in zip(term.references, term.drone_references, strict=True)
        ]
        rows = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=env.device)
        rows = rows.transpose(1, 2)
    return {"p": rows[..., 0:3], "v": rows[..., 3:6], "a": rows[..., 6:9], "tension": rows[..., 9]}


def _convert_rotation_matrix_to_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to their 6D form, the first two columns one after the other.

    Args:
        rotation: Rotation matrices of shape (..., 3, 3).

    Returns:
        6D rotations of shape (..., 6).
    """
    return rotation[..., :2].transpose(-1, -2).flatten(-2)


def _get_attach_points(env: ManagerBasedRLEnv, plant: PlantCfg) -> torch.Tensor:
    """Get where a plant's cables are fixed on the payload, in m in the payload frame.

    Args:
        env: Environment whose device the points go to.
        plant: Plant whose attach points are read, the sim's or the teacher's model.

    Returns:
        Attach points of shape (num_drones, 3).
    """
    points = np.stack([d.attach_point for d in plant.drones])
    return torch.tensor(points, dtype=torch.float32, device=env.device)
