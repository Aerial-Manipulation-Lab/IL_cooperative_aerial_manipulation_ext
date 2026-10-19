"""Flycrane tasks: the payload carried by three drones, the NMPC teacher in the command term."""

import gymnasium as gym

gym.register(
    id="Flycrane-ApproachPose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:ApproachPoseEnvCfg"},
)

gym.register(
    id="Flycrane-FigureEight-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightEnvCfg"},
)

gym.register(
    id="Flycrane-RandomTrajectory-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:RandomTrajectoryEnvCfg"},
)

gym.register(
    id="Flycrane-RandomTrajectoryMediumToFast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:RandomTrajectoryMediumToFastEnvCfg"},
)

gym.register(
    id="Flycrane-FigureEightSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightSlowEnvCfg"},
)

gym.register(
    id="Flycrane-FigureEightMedium-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightMediumEnvCfg"},
)

gym.register(
    id="Flycrane-FigureEightMediumPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightMediumPlusEnvCfg"},
)

gym.register(
    id="Flycrane-FigureEightFast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightFastEnvCfg"},
)

gym.register(
    id="Flycrane-Circle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:CircleEnvCfg"},
)

gym.register(
    id="Flycrane-Square-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:SquareEnvCfg"},
)

gym.register(
    id="Flycrane-Validation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:ValidationEnvCfg"},
)
