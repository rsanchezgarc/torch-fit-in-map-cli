"""Atomic-model reading/writing (via ``mmdf``) and file-level transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    import pandas as pd
    import torch


def read_atoms(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read an atomic model (PDB/mmCIF) into a DataFrame via :func:`mmdf.read`.

    The returned frame has the ``mmdf`` column convention, including ``x``,
    ``y``, ``z`` (Angstroms) and ``element`` (atomic symbol).
    """
    import mmdf  # type: ignore[import]

    return mmdf.read(str(path))


def write_atoms(path: str | os.PathLike[str], atoms: pd.DataFrame) -> None:
    """Write an atoms DataFrame to a PDB/mmCIF file via :func:`mmdf.write`."""
    import mmdf  # type: ignore[import]

    mmdf.write(str(path), atoms)


def pdb_centroid_xyz(atoms: pd.DataFrame) -> tuple[float, float, float]:
    """Return the geometric centroid ``(x, y, z)`` in Angstroms of all atoms."""
    if len(atoms) == 0:
        raise ValueError("atoms DataFrame is empty")
    c = atoms[["x", "y", "z"]].to_numpy().mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def transform_atomic_model(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    rotation_matrix_zyx: torch.Tensor,
    translation_pixels_zyx: torch.Tensor,
    pixel_size: float,
    box_shape: tuple[int, int, int],
    sim_box_size: int | None = None,
    ref_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Read a model, apply an alignment transform to its atoms, and write it out.

    Thin file wrapper around :func:`torch_fit_in_map.transform_atoms`: it reads
    the atoms with ``mmdf``, applies the coordinate transform, and writes the
    result.  The output format is controlled by the ``output_path`` extension
    (``.pdb`` or ``.cif``).

    Parameters
    ----------
    input_path : str or Path
        Input atomic model (PDB or mmCIF).
    output_path : str or Path
        Destination path.
    rotation_matrix_zyx : torch.Tensor
        ``(3, 3)`` rotation matrix in zyx convention from ``AlignmentResult``.
    translation_pixels_zyx : torch.Tensor
        ``(3,)`` translation in zyx pixels from ``AlignmentResult``.
    pixel_size : float
        Voxel size of the reference map in Angstroms.
    box_shape : tuple[int, int, int]
        ``(d, h, w)`` shape of the reference map.
    sim_box_size : int or None
        Cubic box size used during simulation.  Defaults to ``max(box_shape)``.
    ref_origin_xyz : tuple[float, float, float]
        XYZ origin of the reference map in Angstroms (from MRC header).
    """
    from torch_fit_in_map import AlignmentResult, transform_atoms

    atoms = read_atoms(input_path)
    result = AlignmentResult(
        rotation_matrix_zyx,
        translation_pixels_zyx,
        score=float("nan"),
    )
    transformed = transform_atoms(
        atoms,
        result,
        pixel_size=pixel_size,
        box_shape=box_shape,
        sim_box_size=sim_box_size,
        ref_origin_xyz=ref_origin_xyz,
    )
    write_atoms(output_path, transformed)
