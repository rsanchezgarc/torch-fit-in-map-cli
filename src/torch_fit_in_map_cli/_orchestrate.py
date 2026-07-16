"""File-path orchestration: load MRC/PDB, run the alignment, return a result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_fit_in_map import (
    DEFAULT_SIMULATOR,
    AlignmentResult,
    ExhaustiveSearchConfig,
    GradientRefinementConfig,
    crop_or_pad_to_shape,
    fit_map_in_map,
    normalise_voxel_sizes,
)

from ._mrc import load_mrc
from ._pdb import read_atoms

if TYPE_CHECKING:
    import os

    from torch_fit_in_map import DensitySimulator


def _low_pass(
    density: torch.Tensor, pixel_size: float, desired_resolution: float
) -> torch.Tensor:
    """Low-pass filter *density* to *desired_resolution* Angstroms."""
    from torch_fourier_filter.bandpass import low_pass_filter  # type: ignore[import]

    cutoff = pixel_size / desired_resolution  # normalised frequency (0–0.5)
    lp = low_pass_filter(
        cutoff=cutoff,
        falloff=0.02,
        image_shape=density.shape,  # type: ignore[arg-type]
        rfft=True,
        fftshift=False,
        device=density.device,
    )
    ft = torch.fft.rfftn(density, norm="ortho")
    return torch.fft.irfftn(ft * lp, s=density.shape, norm="ortho")


def fit_map_in_map_from_files(
    mobile_path: str | os.PathLike[str],
    reference_path: str | os.PathLike[str],
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask_path: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Load MRC files and fit the mobile volume into the reference.

    Voxel sizes are read from MRC headers; if they differ, the mobile is
    automatically resampled to match the reference via Fourier rescaling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref, ref_px = load_mrc(reference_path)
    mob, mob_px = load_mrc(mobile_path)
    ref, mob, common_px = normalise_voxel_sizes(ref, mob, ref_px, mob_px)

    ref = ref.to(device)
    mob = mob.to(device)

    mask: torch.Tensor | None = None
    if mask_path is not None:
        mask, _ = load_mrc(mask_path)
        mask = mask.to(device)

    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(pixel_size_angstroms=common_px)
    if gradient_config is None and exhaustive_config.pixel_size_angstroms:
        gradient_config = GradientRefinementConfig(
            pixel_size_angstroms=exhaustive_config.pixel_size_angstroms
        )

    return fit_map_in_map(
        mob,
        ref,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )


def fit_map_in_pdb_from_files(
    mobile_map_path: str | os.PathLike[str],
    reference_pdb_path: str | os.PathLike[str],
    pixel_size_angstroms: float | None = None,
    box_size: int | None = None,
    *,
    desired_resolution_angstroms: float | None = None,
    save_simulated: bool = False,
    simulator: DensitySimulator | None = None,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask_path: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Load an MRC map and fit it into the coordinate frame of an atomic model.

    The experimental density (*mobile_map_path*) is the **mobile**; the atoms
    read from *reference_pdb_path* are simulated into the **reference** density.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if simulator is None:
        simulator = DEFAULT_SIMULATOR

    mobile_map, mob_px = load_mrc(mobile_map_path)
    mobile_map = mobile_map.to(device)

    if pixel_size_angstroms is None:
        pixel_size_angstroms = mob_px
    if box_size is None:
        box_size = max(mobile_map.shape[-3:])

    if (
        desired_resolution_angstroms is not None
        and desired_resolution_angstroms < 2.0 * pixel_size_angstroms
    ):
        raise ValueError(
            f"desired_resolution_angstroms ({desired_resolution_angstroms} Å) must be "
            f">= 2 × pixel_size ({2.0 * pixel_size_angstroms} Å)."
        )

    atoms = read_atoms(reference_pdb_path)
    simulated = simulator.simulate(
        atoms=atoms,
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
    )

    if desired_resolution_angstroms is not None:
        simulated = _low_pass(
            simulated, pixel_size_angstroms, desired_resolution_angstroms
        )

    simulated, mobile_map, common_px = normalise_voxel_sizes(
        simulated, mobile_map, pixel_size_angstroms, mob_px
    )
    simulated = crop_or_pad_to_shape(simulated, tuple(mobile_map.shape[-3:]))  # type: ignore[arg-type]

    mask: torch.Tensor | None = None
    if mask_path is not None:
        mask, _ = load_mrc(mask_path)
        mask = mask.to(device)

    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(pixel_size_angstroms=common_px)
    if gradient_config is None and exhaustive_config.pixel_size_angstroms:
        gradient_config = GradientRefinementConfig(
            pixel_size_angstroms=exhaustive_config.pixel_size_angstroms
        )

    result = fit_map_in_map(
        mobile_map,
        simulated,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )

    if save_simulated:
        result.simulated_volume = simulated.cpu()

    return result


def fit_pdb_in_map_from_files(
    mobile_pdb_path: str | os.PathLike[str],
    reference_map_path: str | os.PathLike[str],
    pixel_size_angstroms: float | None = None,
    box_size: int | None = None,
    *,
    desired_resolution_angstroms: float | None = None,
    save_simulated: bool = False,
    simulated_output_path: str | os.PathLike[str] | None = None,
    simulator: DensitySimulator | None = None,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask_path: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Simulate a density from an atomic model and fit it into the target map.

    The atoms read from *mobile_pdb_path* are simulated into the **mobile**
    density; the map at *reference_map_path* is the **reference**.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if simulator is None:
        simulator = DEFAULT_SIMULATOR

    density_map, map_px = load_mrc(reference_map_path)
    density_map = density_map.to(device)

    if pixel_size_angstroms is None:
        pixel_size_angstroms = map_px
    if box_size is None:
        box_size = max(density_map.shape[-3:])

    if (
        desired_resolution_angstroms is not None
        and desired_resolution_angstroms < 2.0 * pixel_size_angstroms
    ):
        raise ValueError(
            f"desired_resolution_angstroms ({desired_resolution_angstroms} Å) must be "
            f">= 2 × pixel_size ({2.0 * pixel_size_angstroms} Å)."
        )

    atoms = read_atoms(mobile_pdb_path)
    simulated = simulator.simulate(
        atoms=atoms,
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
    )

    if desired_resolution_angstroms is not None:
        simulated = _low_pass(
            simulated, pixel_size_angstroms, desired_resolution_angstroms
        )

    density_map, simulated, common_px = normalise_voxel_sizes(
        density_map, simulated, map_px, pixel_size_angstroms
    )
    # After Fourier rescaling the simulated box may have a different shape than the
    # reference map (different box_size, non-cubic reference, rounding in rescale).
    # Crop/pad to match so that FFT cross-correlation is well-defined.
    simulated = crop_or_pad_to_shape(simulated, tuple(density_map.shape[-3:]))  # type: ignore[arg-type]

    mask: torch.Tensor | None = None
    if mask_path is not None:
        mask, _ = load_mrc(mask_path)
        mask = mask.to(device)

    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(pixel_size_angstroms=common_px)

    result = fit_map_in_map(
        simulated,
        density_map,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )

    if save_simulated:
        result.simulated_volume = simulated.cpu()

    if simulated_output_path is not None:
        from ._mrc import save_mrc

        save_mrc(simulated_output_path, simulated, pixel_size=pixel_size_angstroms)

    return result
