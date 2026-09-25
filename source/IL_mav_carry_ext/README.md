# IL MAV Carry

Isaac Lab extension for imitation learning on cooperative aerial manipulation.

It provides the robot assets (`falcon`, `flycart`, `flycrane`, `flycrane_rod`, `flypent`),
the low-level controllers (geometric, INDI, motor model), and the Gym task
registrations under `IL_mav_carry_ext.tasks` for both manager-based and direct MARL
environments.

## Installation

```bash
python -m pip install -e source/IL_mav_carry_ext
```

Inside the Isaac Lab container, use Isaac Sim's interpreter:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e source/IL_mav_carry_ext
```

## Usage

```bash
python scripts/train.py --task <task-id>
python scripts/play.py --task <task-id>
```

Task ids are registered on import of `IL_mav_carry_ext.tasks`.
