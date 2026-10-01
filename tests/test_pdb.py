"""Tests for atomic-model I/O and file-level transforms."""

import numpy as np
import pytest
import torch


def test_read_atoms_columns(tiny_pdb):
    from torch_fit_in_map_cli import read_atoms

    atoms = read_atoms(tiny_pdb)
    assert {"x", "y", "z", "element"} <= set(atoms.columns)
    assert len(atoms) == 8


def test_pdb_centroid(tiny_pdb):
    from torch_fit_in_map_cli import pdb_centroid_xyz, read_atoms

    atoms = read_atoms(tiny_pdb)
    cx, cy, cz = pdb_centroid_xyz(atoms)
    expected = atoms[["x", "y", "z"]].to_numpy().mean(0)
    np.testing.assert_allclose([cx, cy, cz], expected, atol=1e-4)


@pytest.mark.parametrize("suffix", [".pdb", ".ent", ".cif", ".mmcif", ".CIF"])
def test_write_atoms_roundtrip(tiny_pdb, tmp_path, suffix):
    from torch_fit_in_map_cli import read_atoms, write_atoms

    atoms = read_atoms(tiny_pdb)
    out = tmp_path / f"out{suffix}"
    write_atoms(out, atoms)
    reloaded = read_atoms(out)
    assert len(reloaded) == len(atoms)
    for column in ["element", "residue", "chain", "heteroatom_flag"]:
        assert reloaded[column].tolist() == atoms[column].tolist()
    np.testing.assert_allclose(
        reloaded[["x", "y", "z"]].to_numpy(),
        atoms[["x", "y", "z"]].to_numpy(),
        atol=1e-2,
    )


def test_transform_atomic_model_preserves_geometry(tiny_pdb, tmp_path):
    """Identity rotation + zero shift preserves inter-atom distances."""
    from torch_fit_in_map_cli import read_atoms, transform_atomic_model

    out = tmp_path / "moved.pdb"
    transform_atomic_model(
        input_path=tiny_pdb,
        output_path=out,
        rotation_matrix_zyx=torch.eye(3),
        translation_pixels_zyx=torch.zeros(3),
        pixel_size=1.5,
        box_shape=(32, 32, 32),
    )
    a = read_atoms(tiny_pdb)[["x", "y", "z"]].to_numpy()
    b = read_atoms(out)[["x", "y", "z"]].to_numpy()

    def _pdist(p):
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    np.testing.assert_allclose(_pdist(a), _pdist(b), atol=1e-2)
