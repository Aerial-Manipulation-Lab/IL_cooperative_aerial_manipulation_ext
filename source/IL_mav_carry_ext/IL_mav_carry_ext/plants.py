# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical parameters of the plants, as the MPC models them.

Kept apart from `mpc.teacher` on purpose: the observation terms need the same
attach points the MPC uses, and importing them from there would drag the
acados backend into every env that merely observes.
"""

import numpy as np
from python_mpc_cusadi import DroneCfg, PlantCfg

# the flycrane as measured on the hardware, same as examples/run_figure_eight.py
FLYCRANE = PlantCfg(
    load_mass=1.45,
    load_inertia=np.array([0.04, 0.05, 0.08]),
    drones=(
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, 0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([0.26, -0.22, 0.06])),
        DroneCfg(mass=0.6, cable_length=1.0, attach_point=np.array([-0.28, 0.0, 0.06])),
    ),
)

ROPE_END = np.array([0.0, 0.0, -0.03])
"""Where the rope meets each drone, in the drone's body frame: Agilicious' attach_point_drone and p_offset."""

# the flycrane as the sim's USD model builds it, read back from the articulation.
# Differs from the hardware: the attach points are symmetric in x (the hardware
# values sit 1 cm off, most likely measured from an off-centre payload COM), the
# payload is lighter and its inertia ~35% smaller. As in the thesis, a cable is the
# 1 m of rope only: the MPC plans and measures each drone at its rope end, ROPE_END
# below the COM, and the low-level controller lifts the COM back above it.
# Not modelled: the rope's own mass (28 g per cable), and the payload COM sitting
# 2 cm below the odometry link the MPC takes as the payload's state.
FLYCRANE_SIM = PlantCfg(
    load_mass=1.40,
    load_inertia=np.array([0.02625, 0.03792, 0.05892]),
    drones=(
        DroneCfg(mass=0.6168, cable_length=1.0, attach_point=np.array([0.27, 0.22, 0.06])),
        DroneCfg(mass=0.6168, cable_length=1.0, attach_point=np.array([0.27, -0.22, 0.06])),
        DroneCfg(mass=0.6168, cable_length=1.0, attach_point=np.array([-0.27, 0.0, 0.06])),
    ),
)
