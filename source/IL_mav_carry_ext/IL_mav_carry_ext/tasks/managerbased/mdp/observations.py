import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply
from python_mpc_cusadi import Rig

from ....assets import DRONE_IDX, PAYLOAD_IDX
from ....imitation.features import PLAN_PARTS
from ....plants import FLYCRANE_SIM

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
    return matrix_from_quat(drone_quat).flatten(-2)


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
    windows = env.command_manager.get_term(command_name).teacher.windows
    position = _to_tensor(env, [[d.position for d in w.drones] for w in windows])
    velocity = _to_tensor(env, [[d.velocity for d in w.drones] for w in windows])
    acceleration = _to_tensor(env, [[d.acceleration for d in w.drones] for w in windows])
    drones = env.scene[asset_cfg.name].data.body_com_state_w.torch[:, DRONE_IDX]
    position -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    velocity -= drones[:, :, None, 7:10]
    return torch.cat([position, velocity, acceleration], dim=-1).flatten(2)


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
    rig = env.command_manager.get_term(command_name).teacher.rig
    return _get_attach_points(env, rig).expand(env.num_envs, -1, -1)


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
    windows = teacher.windows
    # The payload's reference gets a drone axis, (num_envs, num_nodes, 1, 3), to broadcast over.
    loads = [window.load for window in windows]
    p = _to_tensor(env, [load.position for load in loads])[:, :, None]
    v = _to_tensor(env, [load.velocity for load in loads])[:, :, None]
    a = _to_tensor(env, [load.acceleration for load in loads])[:, :, None]
    w = _to_tensor(env, [load.angular_velocity for load in loads])[:, :, None]
    alpha = _to_tensor(env, [load.angular_acceleration for load in loads])[:, :, None]
    rotation = matrix_from_quat(_to_tensor(env, [load.orientation for load in loads]))[:, :, None]
    # This is a reference, so it uses the teacher's model of the attach points.
    rho = _get_attach_points(env, teacher.rig).expand(env.num_envs, teacher.num_nodes, -1, -1)
    w, alpha = w.expand_as(rho), alpha.expand_as(rho)
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
    windows = env.command_manager.get_term(command_name).teacher.windows
    return _to_tensor(env, [[c.tension for c in w.cables] for w in windows])


"""
Teacher, the student's label.
"""


def teacher_plan_rel_drone(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """What the teacher plans for each drone at each of the MPC's nodes, seen from the drone now.

    This is the student's label. Each node holds, in `PLAN_PARTS` order and along the world axes:

    - the position still to go in m;
    - the velocity still to gain in m/s;
    - the planned acceleration in m/s^2;
    - the planned body rate in rad/s.

    Returns:
        Plan of shape (num_envs, num_drones, num_nodes * PLAN_DIM).
    """
    plans = env.command_manager.get_term(command_name).teacher.plans
    position = _to_tensor(env, [[d.position for d in p.drones] for p in plans])
    velocity = _to_tensor(env, [[d.velocity for d in p.drones] for p in plans])
    acceleration = _to_tensor(env, [[d.acceleration for d in p.drones] for p in plans])
    angular_velocity = _to_tensor(env, [[d.angular_velocity for d in p.drones] for p in plans])
    drones = env.scene["robot"].data.body_com_state_w.torch[:, DRONE_IDX]
    position -= (drones[..., :3] - env.scene.env_origins[:, None])[:, :, None]
    velocity -= drones[:, :, None, 7:10]
    parts = {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "angular_velocity": angular_velocity,
    }
    return torch.cat([parts[name] for name in PLAN_PARTS], dim=-1).flatten(-2)


"""
Helpers.
"""


def _to_tensor(env: ManagerBasedRLEnv, rows: list) -> torch.Tensor:
    """Stack per-env rows of the teacher's trajectories into one tensor on the env's device.

    Args:
        env: Environment whose device the tensor goes to.
        rows: One entry per env, an array or a list of arrays per drone, all of one shape.

    Returns:
        The rows stacked, of shape (num_envs, ...).
    """
    return torch.as_tensor(np.array(rows), dtype=torch.float32, device=env.device)


def _get_attach_points(env: ManagerBasedRLEnv, rig: Rig) -> torch.Tensor:
    """Get where a rig's cables are fixed on the payload, in m in the payload frame.

    Args:
        env: Environment whose device the points go to.
        rig: Rig whose attach points are read, the sim's or the teacher's model.

    Returns:
        Attach points of shape (num_drones, 3).
    """
    points = np.stack([c.attach_point for c in rig.cables])
    return torch.tensor(points, dtype=torch.float32, device=env.device)
