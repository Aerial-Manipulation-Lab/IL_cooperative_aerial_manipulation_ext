from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import (
    configclass,  # Imported explicitly, since isaaclab.utils lazy-exports this name.
)
from isaaclab.utils.noise import GaussianNoiseCfg
from python_mpc_cusadi import Circle, FigureEight, Square

import IL_mav_carry_ext.tasks.managerbased.mdp as mdp
from IL_mav_carry_ext.assets import FLYCRANE_CFG

VALIDATION = {
    "RandomTrajectory": mdp.RandomWalk(
        size=(1.0, 3.0), acceleration=(0.3, 4.5), yaw_rate=(0.0, 0.5)
    ),
    "FigureEight": mdp.RandomFigureEight(
        amplitude=(1.0, 2.5), acceleration=(0.5, 4.5), yaw_rate=(-0.3, 0.3)
    ),
    "Slow": FigureEight(amplitude=2.5, aspect=0.8, period=8 * math.pi),
    "Medium": FigureEight(amplitude=2.5, aspect=0.8, period=4 * math.pi),
    "MediumPlus": FigureEight(amplitude=1.0, aspect=1.0, period=2 * math.pi),
    "Fast": FigureEight(amplitude=2.5, aspect=0.8, period=2 * math.pi),
    "Circle": Circle(),
    "Square": Square(),
}
"""Every reference a student is checked on, by name.

Each task below flies one of them, and the validation task flies them all at once.

- RandomTrajectory is the general training walk. It spans Medium Plus's room for turns (1.2 m)
  to Slow's and Medium's (2.7 m), and from below Slow's 0.5 m/s^2 to past Medium Plus's 4.1, so
  Fast's 8.2 stays unseen. Its heading rate runs from a fixed heading, as the circle has, to past
  the ladder's steady 0.25 rad/s.
- FigureEight is the random figure eight. It spans Medium Plus's size to Medium's, and Slow's
  acceleration to past Medium Plus's, turning either way round.
- Slow, Medium, MediumPlus and Fast are the agility levels of Sun et al. 2025 (Table 1 and S1),
  whose NMPC our teacher is. Each is x = A cos(w s), y = B sin(2 w s), yawing at 0.25 rad/s. Their
  simulated NMPC tracked them to 0.031, 0.067, 0.062 and 0.152 m. The thesis's a2 CSV was Medium,
  and its a4 was Medium Plus with A 1.25 m.
- Circle and Square are the thesis's test shapes (6.4.2).
"""

WALKS_MEDIUM_TO_FAST = (
    mdp.RandomWalk(size=(2.2, 3.0), acceleration=(1.6, 2.5), yaw_rate=(0.0, 0.5)),
    mdp.RandomWalk(size=(1.0, 1.5), acceleration=(3.3, 4.9), yaw_rate=(0.0, 0.5)),
    mdp.RandomWalk(size=(2.2, 3.0), acceleration=(6.6, 9.8), yaw_rate=(0.0, 0.5)),
)
"""Random walks at Sun's Medium, Medium Plus and Fast levels, one level per env in turn.

Each walk's room for turns and peak acceleration are within about 20% of its level. With a multiple
of 3 envs, the three levels get equal shares.

- Medium, 2.7 m and 2.05 m/s^2, peaks offline at about 2.2 m/s and 1.9 m/s^2.
- Medium Plus, 1.2 m and 4.1 m/s^2, peaks offline at about 2.4 m/s and 4.3 m/s^2.
- Fast, 2.7 m and 8.2 m/s^2, peaks offline at about 4.6 m/s and 8.4 m/s^2.

The teacher on its own model flies the hardest of these with up to 20 N of thrust per drone.
"""


"""
Scene.
"""


@configclass
class FlycraneSceneCfg(InteractiveSceneCfg):
    """The flycrane, three drones carrying a payload on cables, above a ground plane."""

    # world
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    distant_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # flycrane
    robot: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="{ENV_REGEX_NS}/flycrane")
    joint_forces = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/flycrane"
    )  # cable forces, for recording


"""
Commands.
"""


@configclass
class ApproachPoseCommandsCfg:
    """A smooth move of the payload to a random goal pose."""

    pose_command = mdp.ApproachPoseCommandCfg(
        debug_vis=True,
        ramp_duration=3.0,  # never more than a gentle move away
        ranges=mdp.ApproachPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            pos_z=(0.5, 1.5),
            roll=(-math.pi / 4, math.pi / 4),
            pitch=(-math.pi / 4, math.pi / 4),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ShapeCommandsCfg:
    """A shape for the payload to fly, from wherever it starts, which each shape task chooses."""

    pose_command = mdp.ShapeTrajectoryCommandCfg(debug_vis=True, shapes=MISSING)


"""
Actions.
"""


@configclass
class ActionsCfg:
    """Each drone's plan relative to itself, which the action term turns into rotor forces.

    The action term adds each drone's position and velocity back to the plan and flies its next
    node with the geometric and INDI controllers.
    """

    low_level_action: mdp.LowLevelActionCfg = mdp.LowLevelActionCfg(
        asset_name="robot", body_name="Falcon.*rotor_.*", debug_vis=False
    )


"""
Observations.
"""


@configclass
class ObservationsCfg:
    """What the student sees, and the teacher's plan it learns."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The student's inputs, one row per drone, of shape (num_envs, num_drones, num_inputs).

        No payload state goes in. Each drone sees:

        - where it should be over the MPC's nodes, and how hard its cable should pull;
        - where its attach point should be, where it is and how fast it moves;
        - how far its attach point is from where it should be;
        - its own orientation and velocities;
        - where its cable is fixed on the payload, which tells the drones apart.

        The references get no noise, since they are commands and not measurements.
        """

        # references
        drone_ref_horizon = ObsTerm(
            func=mdp.drone_ref_horizon, params={"command_name": "pose_command"}
        )
        cable_tension_ref_horizon = ObsTerm(
            func=mdp.cable_tension_ref_horizon, params={"command_name": "pose_command"}
        )
        cable_attach_ref_horizon = ObsTerm(
            func=mdp.cable_attach_ref_horizon, params={"command_name": "pose_command"}
        )

        # cable
        cable_attach_error = ObsTerm(
            func=mdp.cable_attach_error,
            params={"command_name": "pose_command"},
            # 3 mm on the position error and 1 cm/s on the velocity error, as on the measurements
            noise=GaussianNoiseCfg(operation="add", std=torch.tensor([0.003] * 3 + [0.01] * 3)),
        )
        cable_attach_rel_drone = ObsTerm(
            func=mdp.cable_attach_rel_drone, noise=GaussianNoiseCfg(operation="add", std=0.003)
        )
        cable_attach_rel_drone_rate = ObsTerm(
            func=mdp.cable_attach_rel_drone_rate, noise=GaussianNoiseCfg(operation="add", std=0.01)
        )

        # drone
        drone_orientation = ObsTerm(
            func=mdp.drone_orientation, noise=GaussianNoiseCfg(operation="add", std=0.01)
        )
        drone_linear_velocity = ObsTerm(
            func=mdp.drone_linear_velocity, noise=GaussianNoiseCfg(operation="add", std=0.01)
        )
        drone_angular_velocity = ObsTerm(
            func=mdp.drone_angular_velocity, noise=GaussianNoiseCfg(operation="add", std=0.02)
        )
        cable_attach_point = ObsTerm(func=mdp.cable_attach_point)

        def __post_init__(self):
            """Turn the noise off and join the terms into one row per drone."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LabelCfg(ObsGroup):
        """The student's label and the teacher's action, each drone's plan relative to itself.

        Its shape is (num_envs, num_drones, num_nodes * 12).
        """

        teacher_plan_rel_drone = ObsTerm(
            func=mdp.teacher_plan_rel_drone, params={"command_name": "pose_command"}
        )

        def __post_init__(self):
            """Leave the label without noise and join it into one row per drone."""
            self.enable_corruption = False
            self.concatenate_terms = True

    # groups
    policy: PolicyCfg = PolicyCfg()
    label: LabelCfg = LabelCfg()


"""
Events.
"""


@configclass
class EventCfg:
    """What happens to the flycrane at every reset."""

    # Every reset puts the flycrane back in its resting shape, then places it at a random spot and
    # heading, still at rest.
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


"""
Terminations.
"""


@configclass
class TerminationsCfg:
    """When an episode ends, by running out of time or by the flycrane getting into trouble."""

    # The time-out is named success, because Isaac Lab's recorder marks an episode as successful
    # only through a termination of exactly that name. So success means the episode ran to its
    # end without a crash or a failed solve.
    success = DoneTerm(func=mdp.time_out, time_out=True)

    teacher_solve_failed = DoneTerm(func=mdp.teacher_solve_failed)

    # The ground is the one thing a flycrane can hit (self-collisions are off, envs are
    # filtered), so touching it is caught from heights. A contact sensor is not used because its
    # PhysX reading can outlive a reset and end every restart. The drones get 0.2 m, so one tilted
    # onto an arm is caught.
    drone_too_low = DoneTerm(func=mdp.drone_too_low, params={"threshold": 0.2})
    payload_too_low = DoneTerm(func=mdp.payload_too_low, params={"threshold": 0.1})

    # The thresholds are cosines, 0 allowing 90 degrees at the drone and -sqrt(2)/2 allowing 135 at
    # the payload.
    cable_angle_at_drone_too_large = DoneTerm(
        func=mdp.cable_angle_at_drone_too_large, params={"threshold": 0.0}
    )
    cable_angle_at_payload_too_large = DoneTerm(
        func=mdp.cable_angle_at_payload_too_large, params={"threshold": -math.sqrt(2) / 2}
    )

    cables_too_close = DoneTerm(
        func=mdp.cables_too_close, params={"threshold": 0.2, "num_points": 10}
    )

    # The thesis counts an episode that stops with two drones closer than 0.4 m as a failure (6.2).
    drones_too_close = DoneTerm(func=mdp.drones_too_close, params={"threshold": 0.4})

    payload_out_of_bounds = DoneTerm(func=mdp.payload_out_of_bounds, params={"threshold": 5.0})


"""
Environments.
"""


@configclass
class FlycraneEnvCfg(ManagerBasedRLEnvCfg):
    """The flycrane with its MPC teacher, to which each task adds its commands and episode."""

    scene: FlycraneSceneCfg = FlycraneSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Set the physics to 300 Hz and the policy to every third step, 100 Hz."""
        self.decimation = 3
        self.sim.dt = 1 / 300
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.8066)


@configclass
class ApproachPoseEnvCfg(FlycraneEnvCfg):
    """Carry the payload to a random pose."""

    commands: ApproachPoseCommandsCfg = ApproachPoseCommandsCfg()

    def __post_init__(self):
        """Make the episodes 10 s long."""
        super().__post_init__()
        # The 3 s move plus the about 7 s the MPC takes to settle the swinging payload, so an
        # episode holds the whole manoeuvre but little idle hover.
        self.episode_length_s = 10.0


@configclass
class ShapeEnvCfg(FlycraneEnvCfg):
    """Carry the payload along a shape for 40 s, from rest, which each task below chooses."""

    commands: ShapeCommandsCfg = ShapeCommandsCfg()

    def __post_init__(self):
        """Make the episodes 40 s long and widen the box the payload may fly in."""
        super().__post_init__()
        self.episode_length_s = 40.0
        # From a spawn up to 1 m off the env origin, a figure up to 5 m long reaches 6 m, and a walk
        # within 4 m of its start along its own axes, turned any way, reaches 6.7 m.
        self.terminations.payload_out_of_bounds.params["threshold"] = 7.0


@configclass
class FigureEightEnvCfg(ShapeEnvCfg):
    """Carry the payload along a random figure eight."""

    def __post_init__(self):
        """Fly the random figure eight."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["FigureEight"],)


@configclass
class RandomTrajectoryEnvCfg(ShapeEnvCfg):
    """Carry the payload along a random walk, to train for every test reference unseen."""

    def __post_init__(self):
        """Fly the general random walk."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["RandomTrajectory"],)


@configclass
class RandomTrajectoryMediumToFastEnvCfg(ShapeEnvCfg):
    """Carry the payload along random walks at Medium, Medium Plus and Fast, in equal shares."""

    def __post_init__(self):
        """Fly the three walk levels, one per env in turn."""
        super().__post_init__()
        self.commands.pose_command.shapes = WALKS_MEDIUM_TO_FAST


@configclass
class FigureEightSlowEnvCfg(ShapeEnvCfg):
    """Fly Slow, 5 x 4 m with a lap per 25 s, up to 1.2 m/s, 0.5 m/s^2 and 0.3 m/s^3."""

    def __post_init__(self):
        """Fly Slow."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Slow"],)


@configclass
class FigureEightMediumEnvCfg(ShapeEnvCfg):
    """Fly Medium, 5 x 4 m with a lap per 12.6 s, up to 2.4 m/s, 2 m/s^2 and 2 m/s^3."""

    def __post_init__(self):
        """Fly Medium."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Medium"],)


@configclass
class FigureEightMediumPlusEnvCfg(ShapeEnvCfg):
    """Fly Medium Plus, 2 x 2 m with a lap per 6.3 s, up to 2.2 m/s, 4.1 m/s^2 and 8 m/s^3."""

    def __post_init__(self):
        """Fly Medium Plus."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["MediumPlus"],)


@configclass
class FigureEightFastEnvCfg(ShapeEnvCfg):
    """Fly Fast, 5 x 4 m with a lap per 6.3 s, up to 4.7 m/s, 8.2 m/s^2 and 16 m/s^3.

    The payload leans up to 40 degrees.
    """

    def __post_init__(self):
        """Fly Fast."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Fast"],)


@configclass
class CircleEnvCfg(ShapeEnvCfg):
    """Fly the thesis's test circle (6.4.2), radius 2 m with a lap per 20 s, at a fixed heading."""

    def __post_init__(self):
        """Fly the circle."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Circle"],)


@configclass
class SquareEnvCfg(ShapeEnvCfg):
    """Fly the thesis's test square (6.4.2), 2 m sides with 0.5 m corners and a lap per 10 s."""

    def __post_init__(self):
        """Fly the square."""
        super().__post_init__()
        self.commands.pose_command.shapes = (VALIDATION["Square"],)


@configclass
class ValidationEnvCfg(ShapeEnvCfg):
    """Fly every reference in VALIDATION at once, env i the i-th in turn."""

    def __post_init__(self):
        """Fly all the references."""
        super().__post_init__()
        self.commands.pose_command.shapes = tuple(VALIDATION.values())
