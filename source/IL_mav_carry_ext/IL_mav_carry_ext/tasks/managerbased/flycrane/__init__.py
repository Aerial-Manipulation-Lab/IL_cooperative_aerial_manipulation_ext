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
    id="Flycrane-FigureEightA2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightA2EnvCfg"},
)

gym.register(
    id="Flycrane-FigureEightA4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flycrane_env_cfg:FigureEightA4EnvCfg"},
)
