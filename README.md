# torch-fit-in-map-cli

Command-line wrapper for
[`torch-fit-in-map`](https://github.com/teamtomo/teamtomo) — rigid-body volume
alignment for cryo-EM.

The `torch-fit-in-map` algorithm package operates purely on `torch.Tensor`
density maps and `pandas.DataFrame` atom tables. This package owns all the
**file/format** concerns:

- reading/writing MRC density maps (via `mrcfile`),
- reading/writing atomic models as DataFrames (via `mmdf`, so PDB **and** mmCIF
  are supported),
- optional low-pass filtering of simulated densities,
- and the `typer` command-line interfaces.

## Installation

`torch-fit-in-map` and the simulation primitives it uses
(`torch-calculate-electrostatic-potential`, `torch-structure-manipulation`) live
in the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo) and are not yet
on PyPI. Install with [uv](https://docs.astral.sh/uv/), which resolves them from
the monorepo:

```bash
git clone https://github.com/rsanchezgarc/torch-fit-in-map-cli
cd torch-fit-in-map-cli
uv sync
```

Atomic models are simulated as electrostatic potential maps using each atom's
B-factor (`b_isotropic`) from the model file.

## Commands

Three console scripts are installed:

| Command | Purpose |
| --- | --- |
| `torch-fit-in-map REFERENCE MOBILE` | Align a mobile MRC map (or atomic model) onto a reference MRC map. |
| `torch-fit-in-atomic-model REFERENCE MOBILE` | Fit a mobile MRC map into the coordinate frame of a reference atomic model. |
| `torch-simulate-density MODEL -o OUT.mrc` | Simulate an MRC density map from an atomic model. |

### Examples

```bash
# Map-to-map alignment, write result JSON + aligned MRC
torch-fit-in-map reference.mrc mobile.mrc --angular-step 15 \
    --output aligned.mrc --output-json result.json

# Fit an atomic model into a density map (writes a transformed PDB)
torch-fit-in-map reference.mrc model.pdb --output fitted.pdb

# Fit a density map into an atomic-model frame
torch-fit-in-atomic-model model.pdb mobile.mrc --output aligned.mrc

# Simulate a density map from an atomic model
torch-simulate-density model.pdb -o simulated.mrc --pixel-size 1.5 --box-size 128
```

Run any command with `--help` for the full option list (angular step, symmetry,
gradient-refinement settings, multi-GPU `--device`, masking, etc.).

## Programmatic file helpers

The same file-level helpers used by the CLI are importable:

```python
from torch_fit_in_map_cli import (
    fit_map_in_map_from_files,
    fit_map_in_pdb_from_files,
    fit_pdb_in_map_from_files,
    load_mrc,
    save_mrc,
    read_atoms,
    transform_atomic_model,
)
```

For the tensor/DataFrame API (no files), use `torch-fit-in-map` directly.

## Output coordinates and formats

Atomic-model output uses the filename extension: `.pdb`/`.ent` write PDB,
while `.cif`/`.mmcif` write mmCIF.

The top-level JSON rotation and translation fields retain the alignment engine's
**internal box** convention (`transform_frame: "internal_box"`). In particular,
`translation_angstroms_zyx` is a voxel-space shift scaled to Å, not displacement
of the original atomic model in world coordinates. It can be nonzero for a model
that is already correctly placed; subtracting half a voxel is not a general fix.

For atomic-model inputs, JSON also includes `world_transform`, with a forward
XYZ rotation and translation in Å. For column vectors, apply it as:

```python
output_xyz = rotation_matrix_xyz @ input_xyz + translation_angstroms_xyz
```

This transform includes model centering, the box rotation center, cropping,
and the reference map origin. It describes the same coordinates as the written
model (up to file precision).

Simulation includes all atoms read from the input, including waters and
heteroatoms. Remove unwanted atoms before simulation or fitting. Pass `-q` or
`--quiet` to `torch-simulate-density` to suppress progress messages.

## Tests

```bash
uv sync --locked --group test
uv run --no-sync pytest
# Include the transform-recovery integration test:
uv run --no-sync pytest -o addopts=""
```

The lock selects NumPy 2.4.6 on Python 3.11 and 2.5.3 on Python 3.12+.

With NumPy 2.5, mrcfile 1.5.4 emits a dtype-assignment deprecation warning
when reading headers. The test configuration ignores only that specific
third-party warning; all other warnings remain errors.
