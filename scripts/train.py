"""Train an IL policy on a cooperative aerial manipulation task.

Runner-agnostic: everything machine-specific (num_envs, device, paths) arrives
as an argument or config value. Never hardcode num_envs -- it differs between a
48 GB l40 and a 12 GB 3080 Ti.

TODO: port from MARL_cooperative_aerial_manipulation_ext.
"""
