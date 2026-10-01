"""Smoke tests for the typer command-line interfaces."""

import json

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_align_map_to_map(random_map, tmp_path):
    from torch_fit_in_map_cli._cli import align_app

    ref = random_map("ref.mrc")
    mob = random_map("mob.mrc")
    out_json = tmp_path / "result.json"

    result = runner.invoke(
        align_app,
        [
            str(ref),
            str(mob),
            "--angular-step",
            "90",
            "--n-iter",
            "0",
            "--device",
            "cpu",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_json.read_text())
    assert "rotation_matrix_zyx" in data
    assert "translation_pixels_zyx" in data


def test_align_writes_aligned_map(random_map, tmp_path):
    from torch_fit_in_map_cli._cli import align_app

    ref = random_map("ref.mrc")
    mob = random_map("mob.mrc")
    out_mrc = tmp_path / "aligned.mrc"

    result = runner.invoke(
        align_app,
        [
            str(ref),
            str(mob),
            "--angular-step",
            "90",
            "--n-iter",
            "0",
            "--device",
            "cpu",
            "--output",
            str(out_mrc),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_mrc.exists()


def test_align_quiet_requires_output(random_map):
    from torch_fit_in_map_cli._cli import align_app

    ref = random_map("ref.mrc")
    mob = random_map("mob.mrc")
    result = runner.invoke(
        align_app, [str(ref), str(mob), "--quiet", "--device", "cpu"]
    )
    assert result.exit_code == 1
    assert "requires --output" in result.output


@pytest.mark.parametrize("quiet", [False, True])
def test_simulate_frame_and_quiet(tiny_pdb, tmp_path, quiet):
    import numpy as np

    from torch_fit_in_map_cli import load_mrc, pdb_centroid_xyz, read_atoms
    from torch_fit_in_map_cli._cli import simulate_app
    from torch_fit_in_map_cli._mrc import read_mrc_header

    out = tmp_path / "sim.mrc"
    args = [
        str(tiny_pdb),
        "-o",
        str(out),
        "--pixel-size",
        "1.5",
        "--box-size",
        "20",
        "--device",
        "cpu",
        "--desired-resolution",
        "5",
    ]
    if quiet:
        args.append("-q")
    result = runner.invoke(simulate_app, args)
    assert result.exit_code == 0, result.exception
    assert bool(result.output) is not quiet
    shape, px, origin = read_mrc_header(out)
    assert shape == (20, 20, 20)
    assert px == 1.5
    np.testing.assert_allclose(
        np.array(origin) + 9.5 * px, pdb_centroid_xyz(read_atoms(tiny_pdb)), atol=1e-5
    )
    density, _ = load_mrc(out)
    assert density.isfinite().all() and density.abs().sum() > 0


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_atomic_json_world_transform(tiny_pdb, tmp_path, monkeypatch, rotated, suffix):
    import numpy as np
    import torch
    from torch_fit_in_map import AlignmentResult

    from torch_fit_in_map_cli import read_atoms, save_mrc
    from torch_fit_in_map_cli._cli import align_app

    atoms = read_atoms(tiny_pdb)
    xyz = atoms[["x", "y", "z"]].to_numpy()
    centroid = xyz.mean(axis=0)
    px = 1.98
    # A half-voxel box translation can represent exactly zero world displacement.
    origin = centroid - 10 * px
    ref = tmp_path / "ref.mrc"
    save_mrc(ref, torch.zeros(20, 20, 20), px, tuple(origin))
    rotation = (
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        if rotated
        else torch.eye(3)
    )
    result = AlignmentResult(
        rotation,
        torch.full((3,), 0.5),
        1.0,
        translation_angstroms=torch.full((3,), px / 2),
    )
    monkeypatch.setattr(
        "torch_fit_in_map_cli._orchestrate.fit_pdb_in_map_from_files",
        lambda **kwargs: result,
    )
    out = tmp_path / f"out{suffix}"
    out_json = tmp_path / "out.json"
    invocation = runner.invoke(
        align_app,
        [
            str(ref),
            str(tiny_pdb),
            "--device",
            "cpu",
            "--output",
            str(out),
            "--output-json",
            str(out_json),
        ],
    )
    assert invocation.exit_code == 0, invocation.exception
    data = json.loads(out_json.read_text())
    assert data["transform_frame"] == "internal_box"
    np.testing.assert_allclose(data["translation_angstroms_zyx"], px / 2)
    world = data["world_transform"]
    r = np.array(world["rotation_matrix_xyz"])
    t = np.array(world["translation_angstroms_xyz"])
    predicted = xyz @ r.T + t
    expected = (xyz - centroid) @ rotation.numpy()[::-1, ::-1] + centroid
    np.testing.assert_allclose(predicted, expected, atol=1e-5)
    np.testing.assert_allclose(read_atoms(out)[["x", "y", "z"]], predicted, atol=0.001)
    if not rotated:
        np.testing.assert_allclose(t, 0, atol=1e-5)


@pytest.mark.slow
def test_recover_displaced_model(tiny_pdb, tmp_path):
    """Fit a rotated, translated model back into its own simulated map on CPU."""
    import numpy as np

    from torch_fit_in_map_cli import read_atoms, write_atoms
    from torch_fit_in_map_cli._cli import align_app, simulate_app

    ref = tmp_path / "reference.mrc"
    simulation = runner.invoke(
        simulate_app,
        [
            str(tiny_pdb),
            "-o",
            str(ref),
            "--pixel-size",
            "1",
            "--box-size",
            "20",
            "--desired-resolution",
            "3",
            "--device",
            "cpu",
            "-q",
        ],
    )
    assert simulation.exit_code == 0, simulation.exception
    atoms = read_atoms(tiny_pdb)
    original = atoms[["x", "y", "z"]].to_numpy().copy()
    theta = np.deg2rad(50)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    atoms.loc[:, ["x", "y", "z"]] = original @ rotation.T + [8, -6, 4]
    mobile = tmp_path / "displaced.cif"
    write_atoms(mobile, atoms)
    out = tmp_path / "recovered.cif"
    out_json = tmp_path / "fit.json"
    fit = runner.invoke(
        align_app,
        [
            str(ref),
            str(mobile),
            "--output",
            str(out),
            "--output-json",
            str(out_json),
            "--angular-step",
            "30",
            "--n-start",
            "4",
            "--n-iter",
            "100",
            "--device",
            "cpu",
            "--desired-resolution",
            "3",
            "-q",
        ],
    )
    assert fit.exit_code == 0, fit.exception
    recovered = read_atoms(out)[["x", "y", "z"]].to_numpy()
    rmsd = np.sqrt(np.mean(np.sum((recovered - original) ** 2, axis=1)))
    assert rmsd < 0.3
    world = json.loads(out_json.read_text())["world_transform"]
    predicted = (
        atoms[["x", "y", "z"]].to_numpy() @ np.array(world["rotation_matrix_xyz"]).T
        + world["translation_angstroms_xyz"]
    )
    np.testing.assert_allclose(predicted, recovered, atol=1e-4)
