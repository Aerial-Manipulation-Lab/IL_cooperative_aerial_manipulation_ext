from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.configclass import (
    configclass,  # explicit: isaaclab.utils lazy-exports this name and it can be shadowed by the submodule
)
from isaaclab.utils.math import normalize, quat_from_angle_axis

from IL_mav_carry_ext.assets import DRONE_IDX
from IL_mav_carry_ext.controllers import GeometricController, IndiController
from IL_mav_carry_ext.controllers.motor_model import RotorMotor
from IL_mav_carry_ext.imitation.features import PLAN_DIM, PLAN_PARTS

from .marker_utils import (
    ACC_MARKER_CFG,
    DRONE_POS_MARKER_CFG,
    FORCE_MARKER_Z_CFG,
    ORIENTATION_MARKER_CFG,
)


class LowLevelAction(ActionTerm):
    """Flies node 1 of each drone's plan with the geometric and INDI controllers."""

    cfg: LowLevelActionCfg

    def __init__(self, cfg: LowLevelActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # simulation environment
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_ids = self._robot.find_bodies(cfg.body_name)[0]
        self._sim_dt = (
            self._env.sim.get_rendering_dt()
        )  # TODO: rendering dt has to be the same as planner dt

        # configuration
        self.cfg = cfg
        teacher = env.command_manager.get_term(cfg.command_name).teacher
        self._num_drones = teacher.num_drones
        self._num_nodes = teacher.num_nodes

        # buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._forces = torch.zeros(self.num_envs, len(self._body_ids), 3, device=self.device)
        self._prev_forces = torch.zeros(self.num_envs, len(self._body_ids), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        # the setpoint flown this step, node 1 of each drone's plan, in the env frame
        self._setpoint_position = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._setpoint_velocity = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._setpoint_acceleration = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._setpoint_body_rate = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._setpoint_yaw = torch.zeros(self.num_envs, 1, device=self.device)  # heading 0, always

        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(
            self.num_envs, self._num_drones, 4, device=self.device
        )
        self.drone_linear_velocities = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_angular_velocities = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_linear_accelerations = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_angular_accelerations = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        # outer loop controller
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(self.num_envs)
        self._ll_counter = 0
        self._tick = 0

        # inner loop controller
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # motor model
        # experimentally obtained
        self.motor_models = {}
        initial_rpms = [
            torch.tensor([[1441.5819, 1351.1626, 1341.0111, 1428.5597]], device=self.device).repeat(
                self.num_envs, 1
            ),
            torch.tensor([[1377.9199, 1451.8428, 1408.9022, 1329.2014]], device=self.device).repeat(
                self.num_envs, 1
            ),
            torch.tensor([[1281.3964, 1293.0708, 1361.7539, 1347.2434]], device=self.device).repeat(
                self.num_envs, 1
            ),
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self._env.sim.get_physics_dt() * self.cfg.low_level_decimation

        # The spread of the noise on what the pilot measures, at `state_noise_scale` 1.
        self.position_noise_std = 0.003  # 3 mm
        self.orientation_noise_std = 0.01  # ~0.57 degrees in radians
        self.linear_velocity_noise_std = 0.01  # 1 cm/s
        self.angular_velocity_noise_std = 0.02  # 0.02 rad/s
        self.linear_acceleration_noise_std = 0.05  # 0.05 m/s²
        self.angular_acceleration_noise_std = 0.05  # 0.05 rad/s²

        # debug
        if cfg.debug_vis:
            self.drone_positions_debug = torch.zeros(
                self.num_envs, self._num_drones, 3, device=self.device
            )
            self.drone_goals_debug = torch.zeros(
                self.num_envs, self._num_drones, 3, device=self.device
            )
            self.des_acc_debug = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
            self.des_ori_debug = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)

        """
        properties
        """

    @property
    def action_dim(self) -> int:
        return self._num_drones * self._num_nodes * PLAN_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._forces

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # back to the startup state: INDI builds on the current rotor forces, so rotors left saturated
        # by a crash would otherwise flip the drones again right after every restart
        for buf in (
            self._forces,
            self._moments,
            self._prev_forces,
            self._drone_prev_acc,
            self._drone_jerk,
        ):
            buf[env_ids] = 0.0
        for i in range(self._num_drones):
            self.geo_controllers[i].reset(env_ids)
            self._indi_controllers[i].reset(env_ids)
            self.motor_models[i].reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        """Turn each drone's plan into its setpoint: node 1, back in the env frame.

        Args:
            actions: (num_envs, num_drones * num_nodes * PLAN_DIM), each drone's plan in its own
                frame as `teacher_plan_rel_drone` gives it, the drone's position and velocity now
                taken off. They are added back here."""
        self._raw_actions[:] = actions
        drones = self._robot.data.body_com_state_w.torch[:, DRONE_IDX]
        node1 = actions.view(self.num_envs, self._num_drones, self._num_nodes, PLAN_DIM)[:, :, 1]
        self._setpoint_position = (
            node1[..., PLAN_PARTS["position"]]
            + drones[..., :3]
            - self._env.scene.env_origins[:, None]
        )
        self._setpoint_velocity = node1[..., PLAN_PARTS["velocity"]] + drones[..., 7:10]
        self._setpoint_acceleration = node1[..., PLAN_PARTS["acceleration"]]
        self._setpoint_body_rate = node1[..., PLAN_PARTS["angular_velocity"]]
        self._prev_forces = self._forces.clone()
        self._tick = 0

    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            drone_positions = self._env.scene["robot"].data.body_com_state_w.torch[
                :, DRONE_IDX, :3
            ] - self._env.scene.env_origins.unsqueeze(1)
            drone_orientations = self._env.scene["robot"].data.body_com_state_w.torch[
                :, DRONE_IDX, 3:7
            ]
            drone_linear_velocities = self._env.scene["robot"].data.body_com_state_w.torch[
                :, DRONE_IDX, 7:10
            ]
            drone_angular_velocities = self._env.scene["robot"].data.body_com_state_w.torch[
                :, DRONE_IDX, 10:13
            ]
            drone_linear_accelerations = self._env.scene["robot"].data.body_acc_w.torch[
                :, DRONE_IDX, :3
            ]
            drone_angular_accelerations = self._env.scene["robot"].data.body_acc_w.torch[
                :, DRONE_IDX, 3:6
            ]

            self.drone_positions[:] = drone_positions
            self.drone_orientations[:] = drone_orientations
            self.drone_linear_velocities[:] = drone_linear_velocities
            self.drone_angular_velocities[:] = drone_angular_velocities
            self.drone_linear_accelerations[:] = drone_linear_accelerations
            self.drone_angular_accelerations[:] = drone_angular_accelerations
            # Without noise no random number is drawn, so the random starts stay as they were.
            if self.cfg.state_noise_scale > 0.0:
                self._add_state_noise()

            # the setpoint is for the end of this env step; like Agilicious' time sampler, track it at this tick's time
            lead = self._env.step_dt - self._tick * self._env.physics_dt
            for i in range(self._num_drones):
                v, a = self._setpoint_velocity[:, i], self._setpoint_acceleration[:, i]
                setpoint = {
                    "pos": self._setpoint_position[:, i] - v * lead + 0.5 * a * lead**2,
                    "lin_vel": v - a * lead,
                    "lin_acc": a,
                    "body_rates": self._setpoint_body_rate[:, i],
                    "yaw": self._setpoint_yaw,
                }
                drone_states: dict = {}  # dict of tensors
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # calculate current jerk and snap
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (
                    self._sim_dt
                )
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], setpoint
                )
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )

                if self.cfg.debug_vis:
                    self.drone_positions_debug[:, i] = (
                        drone_states["pos"] + self._env.scene.env_origins
                    )
                    self.drone_goals_debug[:, i] = (
                        self._setpoint_position[:, i] + self._env.scene.env_origins
                    )
                    self.des_acc_debug[:, i] = acc_cmd
                    self.des_ori_debug[:, i] = q_cmd
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(
                    target_rpm, self.sampling_time
                )
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1
        self._tick += 1

        # The rotors' wrenches are new every physics step, so they go to the instantaneous composer,
        # which the asset clears after each write. The permanent one is left to lasting wrenches,
        # such as gusts on the payload, which the asset adds in at every write.
        robot = self._env.scene["robot"]
        # apply torques induced by rotors to each body
        robot.instantaneous_wrench_composer.add_forces_and_torques(
            torques=self._moments, body_ids=DRONE_IDX
        )
        # apply forces to each rotor
        robot.instantaneous_wrench_composer.add_forces_and_torques(
            forces=self._forces, body_ids=self._body_ids
        )

    def _add_state_noise(self):
        """Add noise to the drone states the pilot measures, at `state_noise_scale` times each spread."""
        scale = self.cfg.state_noise_scale
        for state, std in (
            (self.drone_positions, self.position_noise_std),
            (self.drone_orientations, self.orientation_noise_std),
            (self.drone_linear_velocities, self.linear_velocity_noise_std),
            (self.drone_angular_velocities, self.angular_velocity_noise_std),
            (self.drone_linear_accelerations, self.linear_acceleration_noise_std),
            (self.drone_angular_accelerations, self.angular_acceleration_noise_std),
        ):
            state += scale * std * torch.randn_like(state)
        # The noisy orientation is a quaternion again only once it has unit length.
        self.drone_orientations[:] = normalize(self.drone_orientations)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "drone_force_z_visualizer"):
                marker_cfg = FORCE_MARKER_Z_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_z_forces"
                self.drone_force_z_visualizer = VisualizationMarkers(marker_cfg)
            self.drone_force_z_visualizer.set_visibility(True)
            if not hasattr(self, "drone_pos_marker"):
                marker_cfg = DRONE_POS_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_pos_markers"
                self.drone_pos_marker = VisualizationMarkers(marker_cfg)
            self.drone_pos_marker.set_visibility(True)
            if not hasattr(self, "acc_marker"):
                marker_cfg = ACC_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_acc"
                self.acc_marker = VisualizationMarkers(marker_cfg)
            self.acc_marker.set_visibility(False)
            if not hasattr(self, "drone_ori_marker"):
                marker_cfg = ORIENTATION_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_ori"
                self.drone_ori_marker = VisualizationMarkers(marker_cfg)
            self.drone_ori_marker.set_visibility(False)

        else:
            if hasattr(self, "drone_force_z_visualizer"):
                self.drone_force_z_visualizer.set_visibility(False)
            if hasattr(self, "drone_pos_marker"):
                self.drone_pos_marker.set_visibility(False)
            if hasattr(self, "acc_marker"):
                self.acc_marker.set_visibility(False)
            if hasattr(self, "drone_ori_marker"):
                self.drone_ori_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # display markers

        # Get drone positions and orientations
        rotor_idx = self._body_ids
        rotor_pos_world_frame = self._robot.data.body_com_state_w.torch[:, rotor_idx, :3].view(
            -1, 3
        )
        rotor_orientation = self._robot.data.body_com_state_w.torch[:, rotor_idx, 3:7].view(-1, 4)

        # marker indices for multiple envs
        marker_indices = [0] * self.num_envs * len(rotor_idx)
        # Rotate the arrow to point in the direction of the force
        zeros = torch.zeros(self.num_envs, 1)
        arrow_rotation = math_utils.quat_from_euler_xyz(
            zeros, (-torch.pi / 2) * torch.ones(self.num_envs, 1), zeros
        )  # rotate -90 degrees around y-axis in the world frame

        # Apply the offset in the drone's local frame by multiplying with drone's orientation
        arrow_rotation_offset = (
            arrow_rotation.repeat(1, len(rotor_idx), 1).to(self.device).view(-1, 4)
        )
        arrow_orientation = math_utils.quat_mul(rotor_orientation, arrow_rotation_offset)

        # scale arrows with applied force
        forces_to_visualize = (self._forces.view(-1, 3)).clone()
        forces_to_visualize[:, :2] += 0.5  # offset to make the forces visible
        forces_to_visualize[:, 2] /= 10  # scale down the forces
        forces_to_visualize[:, [0, 2]] = forces_to_visualize[
            :, [2, 0]
        ]  # swap around because the arrow prim is oriented in x direction
        # drone_pos_world_frame[:,2] += forces_to_visualize[:,2]/2 # offset because cylinder is drawn from the center TODO
        self.drone_force_z_visualizer.visualize(
            rotor_pos_world_frame, arrow_orientation, forces_to_visualize, marker_indices
        )

        # drone positions
        positions = torch.cat(
            (self.drone_positions_debug, self.drone_goals_debug), dim=0
        )  # visualize the payload positions in world frame
        marker_idx = (
            [0] * self.num_envs * self._num_drones
            + [1] * self.num_envs
            + [2] * self.num_envs
            + [3] * self.num_envs
        )
        self.drone_pos_marker.visualize(
            translations=positions.view(-1, 3), marker_indices=marker_idx
        )

        # drone desired accelerations

        # Normalize the desired direction vector (which represents the direction)
        acc_orientation_axis = normalize(self.des_acc_debug)
        # Define the default x-axis (the direction the arrow marker points to by default)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, self._num_drones, 1
        )
        # Compute the dot product between x-axis and desired direction to check alignment
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        # Flip the desired direction if the dot product is negative (indicating opposite direction)
        mask = (cos_angle.view(-1, 1) < 0).squeeze()
        # acc_orientation_axis = torch.where(mask.squeeze(), -acc_orientation_axis.view(-1,3), acc_orientation_axis.view(-1,3))
        acc_orientation_axis.view(-1, 3)[mask] = -acc_orientation_axis.view(-1, 3)[mask]
        # Compute the axis of rotation (cross product between x-axis and desired direction)
        rotation_axis = torch.linalg.cross(x_axis, acc_orientation_axis)
        # Compute the angle between x-axis and desired direction using dot product
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # Clamp to avoid numerical issues
        # Handle cases where the vectors are parallel (no rotation needed)
        rotation_axis = torch.where(
            torch.norm(rotation_axis, dim=-1, keepdim=True) < 1e-6,  # Check if parallel
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # Default to any orthogonal axis
            normalize(rotation_axis),
        )
        # Compute the quaternion from the angle-axis representation
        acc_orientation = quat_from_angle_axis(angle.view(-1), rotation_axis.view(-1, 3)).view(
            -1, 4
        )

        # Visualize the arrow marker with the new orientation
        self.acc_marker.visualize(
            self.drone_positions_debug.view(-1, 3),
            acc_orientation.view(-1, 4),
            self.des_acc_debug.view(-1, 3) / 5,
            marker_indices=[0] * self.num_envs * self._num_drones,
        )

        # Visualize the desired orientation of the drone
        self.drone_ori_marker.visualize(
            self.drone_positions_debug.view(-1, 3),
            self.des_ori_debug.view(-1, 4),
            marker_indices=[0] * self.num_envs * self._num_drones,
        )


@configclass
class LowLevelActionCfg(ActionTermCfg):
    """Configuration for the low level action term."""

    class_type: type[ActionTerm] = LowLevelAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the scene for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset on which the forces are applied: Falcon.*base_link or Falcon.*rotor*."""
    command_name: str = "pose_command"
    """Command whose teacher sets the plan's shape, and whose plan the teacher's actions are."""
    low_level_decimation: int = 1
    """Decimation factor for the low level action term."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
    state_noise_scale: float = 0.0
    """Noise on the drone states the pilot measures, times the spreads set in the term, 0 for none."""
