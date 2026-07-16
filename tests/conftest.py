"""Shared fixtures for torch-fit-in-map-cli tests."""

from pathlib import Path

import numpy as np
import pytest

# A minimal but valid PDB with a handful of atoms.
_TINY_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -0.773   1.204  1.00  0.00           C
ATOM      6  N   GLY A   2       3.332   1.552   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.999   2.845   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.505   2.687   0.000  1.00  0.00           C
END
"""


def _write_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size


@pytest.fixture
def tiny_pdb(tmp_path) -> Path:
    p = tmp_path / "tiny.pdb"
    p.write_text(_TINY_PDB)
    return p


@pytest.fixture
def random_map(tmp_path):
    """Return a factory that writes a random MRC and returns its path."""

    def _make(
        name: str = "map.mrc", shape=(20, 20, 20), voxel_size: float = 1.0
    ) -> Path:
        data = np.random.default_rng(0).random(shape).astype(np.float32)
        path = tmp_path / name
        _write_mrc(path, data, voxel_size)
        return path

    return _make
