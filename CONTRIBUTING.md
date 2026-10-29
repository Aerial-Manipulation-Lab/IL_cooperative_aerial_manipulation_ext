# Contributing to `IL_cooperative_aerial_manipulation_ext`

Code style follows [Isaac Lab's contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html):
Google Python Style Guide, PEP-8/484/585, Google-style docstrings. Line length is 100.

## Naming

| Thing | Convention | Example |
|---|---|---|
| Functions | `verb_object`, leading `_` if private | `get_payload_pose`, `_compute_attach_points` |
| Verbs | `get_` reads from the scene, `compute_` does maths, `convert_` changes a format | `convert_rotation_matrix_to_rot6d` |
| Observation terms | nouns, subject first (`payload_`, `drone_`, `cable_`, `teacher_`), spelled out, singular; `ref` and `rel` stay short | `payload_linear_velocity`, `cable_attach_ref_horizon` |
| Stateful observation terms | `ManagerTermBase` classes named like the function terms | `cable_attach_rel_drone_rate` |
| Constants | `ALL_CAPS` with a docstring underneath | `SOLVER_OPENMP_THREADS` |
| Configs | `*Cfg` configclasses | `ShapeTrajectoryCommandCfg` |
| Classes | `CapWords` | `MpcTeacher`, `LowLevelAction` |
| Fields and attributes | nouns, for what they hold | `num_drones`, `node_offsets` |
| Self-free helpers | private module-level `_functions` at the bottom of the file | `_convert_rotation_matrix_to_rot6d` |

Frames and units go in docstrings, not in names. Metrics are the exception: their unit is the
name's suffix (`pos_rmse_m`), which `TextReporter` prints.

## Docstrings

- Google style (Napoleon), on every class and public function; modules have
  no docstrings.
- Write for a reader new to the project. Explain in whole sentences what the
  thing is and what it is for, not as a packed list of words.
- Make a docstring as long as the thing needs and no longer. A simple term
  stays at its summary; a term that packs several quantities gets a short
  explanation.
- Use the project's words (reference, the MPC's nodes, teacher plan) as the
  glossary defines them, without re-explaining them in every docstring.
- When a docstring lists several things, put each on its own `-` bullet.
- Name things fully ("a term shared by all drones", not "a term").
- One summary sentence plus `Args`/`Returns`/`Raises`. That is the contract;
  design reasons belong in comments.
- Observation terms take the short form, one summary sentence plus `Returns`
  with the shape. Their arguments (`env`, `asset_cfg`, `command_name`) are the
  same everywhere and are not listed.
- Type hints in signatures only, never repeated in the docstring.
- Document array shapes as tuples, `(num_envs, num_drones, 3)`, in
  `Args`/`Returns`.
- State the frame and the unit of every quantity.
- No `-> None` in prose.

## Comments

- Docstrings state the contract; the *why* goes in inline `#` comments at the
  line where it bites.
- Comments are full sentences explaining what the line does and why the code
  is structured this way -- readable on their own, no telegraphese
  ("slow: ..." is not a comment).
- No explanations of how other libraries work internally; third-party
  behavior appears only as the shortest clause that justifies a structure.
- Don't comment what a well-known construct does; only why it's there.
  Good: `# @runtime_checkable is used for testing`. One short line, no
  trailing explanation of the mechanism.
- No `:` in comments or docstring prose. Use parentheses, commas, or a new
  sentence instead. Google section headers (`Args:`, `Returns:`) and `Args`
  entry lines are exempt.

## File and class skeleton

Module: imports, constants, public functions, classes, then private functions.
Private module-level functions always sit at the bottom of the file, used by
the class above them or not. Constants built from the module's own classes
come right after those classes. Groups of functions may be introduced by a
capitalized bare string, as sections in a class are:

```python
"""
Payload.
"""
```

Class body, in order: constants, `ClassVar` class variables, `__init__`,
`__repr__`/`__str__`, then sections introduced by a capitalized bare string:

```python
"""
Properties.
"""
```

Properties, public operations (in call order), public class/static methods,
private members.

Config fields carry a short comment on their line (`N: int = 20  # shooting nodes`);
groups of fields are introduced by a short lowercase comment above them
(`# cost weights`).

## Scripts

The scripts in `scripts/` are entry points, with their own rules.

- A docstring at the top saying what the script does, with an example command.
- Arguments only through argparse, with a `help` on every flag.
- No helpers or classes; everything in `main()`, top to bottom.
- Reporting only through `TextReporter` and `TensorBoardReporter`.
- Comments may be shorter than in the package.

## Commits and branches

- Commit messages are a short title, then bullets saying what changed. No
  attribution lines.
- Branches are named `experiment/<what>` for experiments, `feature/<what>` or
  `cleanup` otherwise.

## Tools

No enforcement beyond the existing pre-commit hook (ruff check and format,
config in `ruff.toml`); run manually:

```bash
ruff check scripts source
ruff format scripts source
```
