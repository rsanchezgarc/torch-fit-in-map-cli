"""Smoke tests for the typer command-line interfaces."""

import json

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
