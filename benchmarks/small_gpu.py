"""Benchmark small CLI simulation/fitting jobs on CPU and CUDA.

Run from the repository root with:
    python benchmarks/small_gpu.py --output /tmp/small-gpu-results.json
Uses the eight-atom test fixture; timings do not predict full-protein throughput.
"""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from typer.testing import CliRunner

from torch_fit_in_map_cli import load_mrc, read_atoms, write_atoms
from torch_fit_in_map_cli._cli import align_app, simulate_app


def main() -> None:
    """Run identical warmed-up CLI workloads and save timing/accuracy results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    # Leave headroom for the display and allocations outside the PyTorch allocator.
    torch.cuda.set_per_process_memory_fraction(0.5)
    runner = CliRunner()
    records = []

    def invoke(app: object, argv: list[str], device: str) -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = runner.invoke(app, argv)
        if result.exit_code:
            raise RuntimeError(result.output) from result.exception
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    fixture = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/conftest.py")
    )
    with tempfile.TemporaryDirectory(prefix="fit-gpu-benchmark-") as directory:
        root = Path(directory)
        model = root / "original.pdb"
        model.write_text(fixture["_TINY_PDB"])
        atoms = read_atoms(model)
        xyz = atoms[["x", "y", "z"]].to_numpy().copy()
        theta = np.deg2rad(50)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        atoms.loc[:, ["x", "y", "z"]] = xyz @ rotation.T + [8, -6, 4]
        mobile = root / "mobile.cif"
        write_atoms(mobile, atoms)

        for box in [20, 64]:
            reference = root / f"reference-{box}.mrc"
            common_sim = [
                str(model),
                "--pixel-size",
                "1",
                "--box-size",
                str(box),
                "--desired-resolution",
                "3",
                "-q",
            ]
            invoke(
                simulate_app,
                [*common_sim, "-o", str(reference), "--device", "cpu"],
                "cpu",
            )
            ref, _ = load_mrc(reference)
            for device in ["cpu", "cuda"]:
                simulated = root / f"simulated-{box}-{device}.mrc"
                fitted = root / f"fitted-{box}-{device}.cif"
                json_path = root / "fit.json"
                sim_args = [*common_sim, "-o", str(simulated), "--device", device]
                fit_args = [
                    str(reference),
                    str(mobile),
                    "--output",
                    str(fitted),
                    "--output-json",
                    str(json_path),
                    "--angular-step",
                    "30",
                    "--n-start",
                    "4",
                    "--n-iter",
                    "100",
                    "--device",
                    device,
                    "--desired-resolution",
                    "3",
                    "-q",
                ]
                # Full untimed warm-up includes CUDA context, FFT plans and kernels.
                invoke(simulate_app, sim_args, device)
                invoke(align_app, fit_args, device)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sim_times, fit_times, rmsds = [], [], []
                for _ in range(args.repeats):
                    sim_times.append(invoke(simulate_app, sim_args, device))
                    fit_times.append(invoke(align_app, fit_args, device))
                    recovered = read_atoms(fitted)[["x", "y", "z"]].to_numpy()
                    rmsds.append(
                        float(np.sqrt(np.mean(np.sum((recovered - xyz) ** 2, axis=1))))
                    )
                density, _ = load_mrc(simulated)
                relative_error = float(
                    torch.linalg.vector_norm(density - ref)
                    / torch.linalg.vector_norm(ref)
                )
                record = {
                    "box": box,
                    "atoms": len(atoms),
                    "device": device,
                    "simulation_seconds": sim_times,
                    "fit_seconds": fit_times,
                    "median_simulation_seconds": statistics.median(sim_times),
                    "median_fit_seconds": statistics.median(fit_times),
                    "rmsd_angstroms": rmsds,
                    "score": json.loads(json_path.read_text())["score"],
                    "simulation_relative_l2_error_vs_cpu": relative_error,
                    "peak_allocated_mib": (
                        torch.cuda.max_memory_allocated() / 2**20
                        if device == "cuda"
                        else None
                    ),
                    "peak_reserved_mib": (
                        torch.cuda.max_memory_reserved() / 2**20
                        if device == "cuda"
                        else None
                    ),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                if max(rmsds) >= 0.3 or relative_error >= 1e-4:
                    raise AssertionError(f"Accuracy check failed: {record}")
    args.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cpu_threads": torch.get_num_threads(),
                "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory
                / 2**20,
                "warmups_per_case": 1,
                "repeats": args.repeats,
                "pixel_size_angstroms": 1,
                "resolution_angstroms": 3,
                "angular_step_degrees": 30,
                "n_start": 4,
                "n_iter": 100,
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
