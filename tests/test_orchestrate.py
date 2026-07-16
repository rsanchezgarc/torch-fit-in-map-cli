"""Tests for the file-path orchestration wrappers."""

import sys

import pytest
import torch


def test_fit_map_in_map_from_files_same_map(random_map):
    """Fitting identical maps from files recovers a near-identity rotation."""
    from torch_fit_in_map import ExhaustiveSearchConfig

    from torch_fit_in_map_cli import fit_map_in_map_from_files

    ref = random_map("ref.mrc")
    mob = random_map("mob.mrc")  # same rng seed → identical content

    result = fit_map_in_map_from_files(
        mob,
        ref,
        exhaustive_config=ExhaustiveSearchConfig(angular_step_degrees=30.0),
        gradient_config=None,
        verbose=False,
    )
    assert torch.allclose(result.rotation_matrix.cpu(), torch.eye(3), atol=0.15)


def test_fit_map_in_map_from_files_voxel_mismatch(random_map):
    """Files with different voxel sizes are rescaled and still produce a score."""
    from torch_fit_in_map import ExhaustiveSearchConfig

    from torch_fit_in_map_cli import fit_map_in_map_from_files

    ref = random_map("ref.mrc", voxel_size=1.0)
    mob = random_map("mob.mrc", voxel_size=2.0)

    result = fit_map_in_map_from_files(
        mob,
        ref,
        exhaustive_config=ExhaustiveSearchConfig(angular_step_degrees=30.0),
        gradient_config=None,
        verbose=False,
    )
    assert isinstance(result.score, float)


def test_fit_pdb_in_map_from_files_raises_without_espcalculator(
    tiny_pdb, random_map, monkeypatch
):
    """The default simulator surfaces a helpful error when espcalculator is missing."""
    from torch_fit_in_map_cli import fit_pdb_in_map_from_files

    map_path = random_map("map.mrc")
    monkeypatch.setitem(sys.modules, "espcalculator", None)

    pattern = r"espcalculator|torch-calculate"
    with pytest.raises((ImportError, Exception), match=pattern):
        fit_pdb_in_map_from_files(
            mobile_pdb_path=tiny_pdb,
            reference_map_path=map_path,
            pixel_size_angstroms=1.0,
            box_size=20,
            verbose=False,
        )
