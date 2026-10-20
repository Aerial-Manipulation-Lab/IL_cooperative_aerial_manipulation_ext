from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import (
    configclass,  # explicit: isaaclab.utils lazy-exports this name and it can be shadowed by the submodule
)
from isaaclab.utils.noise import GaussianNoiseCfg
from python_mpc_cusadi import Circle, FigureEight, Square

import IL_mav_carry_ext.tasks.managerbased.mdp as mdp
from IL_mav_carry_ext.assets import FLYCRANE_CFG


# Isaac Lab 3.0 folded AdditiveGaussianNoiseCfg into GaussianNoiseCfg(operation="add")
def Gnoise(**kwargs):
    return GaussianNoiseCfg(operation="add", **kwargs)


# Define the scene configuration


@configclass
class CarryingSceneCfg(InteractiveSceneCfg):
    """Configuration for multi-drone lifting system"""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Drones
    robot: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="{ENV_REGEX_NS}/flycrane")

    joint_forces = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/flycrane")


# MDP settings


@configclass
class ApproachPoseCommandsCfg:
    """A min-jerk approach to a random payload pose."""

    pose_command = mdp.ApproachPoseCommandCfg(
        asset_name="robot",
        body_name="load_odometry_sensor_link",
        debug_vis=True,
        ramp_duration=3.0,  # min-jerk ramp: never more than a gentle move away
        ranges=mdp.ApproachPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            pos_z=(0.5, 1.5),
            roll=(-math.pi / 4, math.pi / 4),
            pitch=(-math.pi / 4, math.pi / 4),
            yaw=(-math.pi, math.pi),
        ),
    )


# Every reference a student is checked on, by name. Each task below flies one of them, validation all at once.
# The ladder is the agility levels of Sun et al. 2025 (Table 1, Table S1), whose NMPC our teacher is: x = A cos(w s),
# y = B sin(2 w s), yawing at 0.25 rad/s, a lap per 2 pi / w of the clock. Peaks below are on our clock; the paper's
# simulated NMPC tracked them to 0.031, 0.067, 0.062 and 0.152 m. The thesis's a2 CSV was Medium, its a4 Medium Plus
# with A 1.25 m.
VALIDATION = {
    # the training walk: Medium Plus's 1.2 m to Slow's and Medium's 2.7, below Slow's 0.5 m/s^2 to past Medium Plus's
    # 4.1 (Fast's 8.2 stays unseen), from a fixed heading, as the circle's, past the ladder's steady 0.25 rad/s
    "RandomTrajectory": mdp.RandomWalk(size=(1.0, 3.0), acceleration=(0.3, 4.5), yaw_rate=(0.0, 0.5)),
    # Medium Plus's size to Medium's, Slow's 0.5 m/s^2 to past Medium Plus's 4.1, both ways round
    "FigureEight": mdp.RandomFigureEight(amplitude=(1.0, 2.5), acceleration=(0.5, 4.5), yaw_rate=(-0.3, 0.3)),
    "Slow": FigureEight(amplitude=2.5, aspect=0.8, period=8 * math.pi),
    "Medium": FigureEight(amplitude=2.5, aspect=0.8, period=4 * math.pi),
    "MediumPlus": FigureEight(amplitude=1.0, aspect=1.0, period=2 * math.pi),
    "Fast": FigureEight(amplitude=2.5, aspect=0.8, period=2 * math.pi),
    "Circle": Circle(),
    "Square": Square(),
}


# Random walks around Sun's Medium, Medium Plus and Fast levels (size v^2/a and peak acceleration within about 20% of
# each), one band per env, round robin: with a multiple of 3 envs, equal shares. Offline (2026-10-19) they peak at
# 2.2 / 2.4 / 4.6 m/s and 1.9 / 4.3 / 8.4 m/s^2, the own-model teacher flying the hard ends with up to 20 N of thrust.
WALKS_MEDIUM_TO_FAST = (
    mdp.RandomWalk(size=(2.2, 3.0), acceleration=(1.6, 2.5), yaw_rate=(0.0, 0.5)),  # Medium: 2.7 m, 2.05 m/s^2
    mdp.RandomWalk(size=(1.0, 1.5), acceleration=(3.3, 4.9), yaw_rate=(0.0, 0.5)),  # Medium Plus: 1.2 m, 4.1 m/s^2
    mdp.RandomWalk(size=(2.2, 3.0), acceleration=(6.6, 9.8), yaw_rate=(0.0, 0.5)),  # Fast: 2.7 m, 8.2 m/s^2
)


@configclass
class FigureEightCommandsCfg:
    """A random figure eight, from wherever the payload starts; the other shape tasks swap in their own `shapes`."""

    pose_command = mdp.ShapeTrajectoryCommandCfg(
        asset_name="robot",
        body_name="load_odometry_sensor_link",
        debug_vis=True,
        shapes=(VALIDATION["FigureEight"],),
    )


@configclass
class ActionsCfg:
    """Actions for the flycrane: each drone's plan in its own frame, as `teacher_plan_rel_drone` gives it.

    The action term adds each drone's position and velocity back and flies node 1 with the
    geometric and INDI controllers."""

    low_level_action: mdp.LowLevelActionCfg = mdp.LowLevelActionCfg(
        asset_name="robot",
        body_name="Falcon.*rotor_.*",
        debug_vis=False,
        control_mode="geometric",  # node 1's p, v, a, w per drone
        feedforward_command="pose_command",  # the plan is a correction on each drone's reference
    )


@configclass
class ObservationsCfg:
    """Observations for the flycrane."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The uniform student's input: (num_envs, num_drones, input_dim), one row per drone.

        The baseline's inputs, payload state included, plus where the drone should be and how hard its cable should
        pull (the teacher's own drone reference, from the payload reference alone). The payload terms and the
        payload reference are the same for every drone, the rest each drone's own; the attach point tells the
        drones apart. No noise on the references: they are commands, not measurements.
        """

        payload_orientation = ObsTerm(
            func=mdp.copy_to_each_drone, params={"term": mdp.payload_orientation}, noise=Gnoise(std=0.01)
        )
        payload_linear_velocities = ObsTerm(
            func=mdp.copy_to_each_drone, params={"term": mdp.payload_linear_velocities}, noise=Gnoise(std=0.01)
        )
        payload_angular_velocities = ObsTerm(
            func=mdp.copy_to_each_drone, params={"term": mdp.payload_angular_velocities}, noise=Gnoise(std=0.02)
        )
        payload_ref_horizon = ObsTerm(
            func=mdp.copy_to_each_drone,
            params={"term": mdp.payload_ref_horizon, "params": {"command_name": "pose_command"}},
        )
        drone_feedforward_horizon = ObsTerm(func=mdp.drone_feedforward_horizon, params={"command_name": "pose_command"})
        drone_tension_reference = ObsTerm(func=mdp.drone_tension_reference, params={"command_name": "pose_command"})

        drone_pos_rel_payload = ObsTerm(func=mdp.payload_drone_rpos, noise=Gnoise(std=0.003))
        drone_orientations = ObsTerm(func=mdp.drone_orientations, noise=Gnoise(std=0.01))
        drone_linear_velocities = ObsTerm(func=mdp.drone_linear_velocities, noise=Gnoise(std=0.01))
        drone_angular_velocities = ObsTerm(func=mdp.drone_angular_velocities, noise=Gnoise(std=0.02))
        drone_cable_dir = ObsTerm(func=mdp.drone_cable_dir, noise=Gnoise(std=0.01))
        drone_attach_point = ObsTerm(func=mdp.drone_attach_point)

        def __post_init__(self):
            self.enable_corruption = False  # for adding noise to the observations
            self.concatenate_terms = True

    @configclass
    class LabelCfg(ObsGroup):
        """The student's label and the teacher's action: each drone's plan in its own frame,
        (num_envs, num_drones, num_nodes * 12)."""

        teacher_plan_residual = ObsTerm(func=mdp.teacher_plan_residual, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    label: LabelCfg = LabelCfg()


@configclass
class EventCfg:
    """Events for the flycrane.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Terminal conditions for the flycrane.

    When the payload reaches a certain height, etc."""

    # end when sim times out. Named `success` because Isaac Lab's recorder marks
    # an episode successful only through a termination term of exactly that
    # name (and overwrites any flag set by hand right before it exports). So
    # "success" means the episode ran to its end without crashing or a failed
    # solve; whether it also ended close to the goal is for the dataset loader
    # to decide from the recorded states.
    success = DoneTerm(func=mdp.time_out, time_out=True)

    mpc_failed = DoneTerm(func=mdp.mpc_failed)

    # end on touching the ground, the one thing a flycrane can hit (self-collisions are off, envs filtered).
    # From heights, not a contact sensor: its PhysX reading can outlive a reset and end every restart.
    # 0.2 m for the drones, so one tilted onto an arm is caught too
    falcon_fly_low = DoneTerm(func=mdp.falcon_fly_low, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2})
    payload_fly_low = DoneTerm(
        func=mdp.payload_fly_low, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.1}
    )

    angle_drones_cable = DoneTerm(
        func=mdp.cable_angle_drones_cos, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.0}
    )
    angle_load_cable = DoneTerm(
        func=mdp.cable_angle_payload_cos, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": -math.sqrt(2) / 2}
    )

    cables_collide = DoneTerm(
        func=mdp.cable_collision, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2, "num_points": 10}
    )

    # the thesis counts an episode that stops with two drones closer than 0.4 m as a failure (6.2)
    drones_collide = DoneTerm(func=mdp.drone_collision, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4})

    bounding_box = DoneTerm(func=mdp.bounding_box, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 5.0})


@configclass
class FlycraneEnvCfg(ManagerBasedRLEnvCfg):
    """The flycrane and its NMPC teacher; each task adds its `commands` and episode length."""

    scene: CarryingSceneCfg = CarryingSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 3
        # simulation settings
        self.sim.dt = 0.0033333333333333335
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.8066)


@configclass
class ApproachPoseEnvCfg(FlycraneEnvCfg):
    """Carry the payload to a random pose."""

    commands: ApproachPoseCommandsCfg = ApproachPoseCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # the 3 s ramp plus the ~7 s the MPC takes to settle the swinging payload
        # (measured with check_student_obs.py --episode_length 20: flat from ~10 s),
        # so episodes hold the whole manoeuvre but little idle hover
        self.episode_length_s = 10.0


@configclass
class FigureEightEnvCfg(FlycraneEnvCfg):
    """Carry the payload along a random figure eight."""

    commands: FigureEightCommandsCfg = FigureEightCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0  # the thesis's figure eight, rest to rest
        # from a spawn up to 1 m off the env origin: a figure up to 5 m long reaches 6 m, a walk within 4 m of its
        # start along its own x and y, turned any way, 6.7 m
        self.terminations.bounding_box.params["threshold"] = 7.0


@configclass
class RandomTrajectoryEnvCfg(FigureEightEnvCfg):
    """Carry the payload along a random walk, rest to rest: training for every test reference without flying any of
    their shapes."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["RandomTrajectory"],)


@configclass
class RandomTrajectoryMediumToFastEnvCfg(FigureEightEnvCfg):
    """Carry the payload along random walks at Medium, Medium Plus and Fast difficulty, in equal shares with a multiple
    of 3 envs: training up to the hardest test reference, still without flying any of their shapes."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = WALKS_MEDIUM_TO_FAST


@configclass
class FigureEightSlowEnvCfg(FigureEightEnvCfg):
    """Slow: 5 x 4 m, a lap per 25 s, up to 1.2 m/s, 0.5 m/s^2 and 0.3 m/s^3."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Slow"],)


@configclass
class FigureEightMediumEnvCfg(FigureEightEnvCfg):
    """Medium: 5 x 4 m, a lap per 12.6 s, up to 2.4 m/s, 2 m/s^2 and 2 m/s^3."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Medium"],)


@configclass
class FigureEightMediumPlusEnvCfg(FigureEightEnvCfg):
    """Medium Plus: 2 x 2 m, a lap per 6.3 s, up to 2.2 m/s, 4.1 m/s^2 and 8 m/s^3."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["MediumPlus"],)


@configclass
class FigureEightFastEnvCfg(FigureEightEnvCfg):
    """Fast: 5 x 4 m, a lap per 6.3 s, up to 4.7 m/s, 8.2 m/s^2 and 16 m/s^3, leaning 40 degrees."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Fast"],)


@configclass
class CircleEnvCfg(FigureEightEnvCfg):
    """The thesis's test circle (6.4.2): radius 2 m, a lap per 20 s, fixed heading."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Circle"],)


@configclass
class SquareEnvCfg(FigureEightEnvCfg):
    """The thesis's test square (6.4.2): 2 m sides, 0.5 m corners, a lap per 10 s, fixed heading."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Square"],)


@configclass
class ValidationEnvCfg(FigureEightEnvCfg):
    """Every reference in VALIDATION at once: env i flies the i-th, round robin."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_command.shapes = tuple(VALIDATION.values())
