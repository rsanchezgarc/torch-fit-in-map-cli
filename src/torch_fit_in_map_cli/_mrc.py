"""MRC density-map reading and writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import os


def load_mrc(path: str | os.PathLike[str]) -> tuple[torch.Tensor, float]:
    """Load an MRC file and return ``(data_tensor, pixel_size_angstroms)``."""
    import mrcfile  # type: ignore[import]

    with mrcfile.open(str(path), mode="r") as mrc:
        data = torch.from_numpy(mrc.data.copy()).float()
        px = float(mrc.voxel_size.x)
    if px == 0.0:
        px = 1.0  # fallback if header has no pixel size
    return data, px


def read_mrc_header(
    path: str | os.PathLike[str],
) -> tuple[tuple[int, int, int], float, tuple[float, float, float]]:
    """Return ``(shape_dhw, pixel_size_angstroms, origin_xyz_angstroms)`` from a header.

    Origin is read from the MRC2014 ``origin`` field.  If that field is all
    zeros, the older ``nxstart/nystart/nzstart`` convention is used as a
    fallback (``n*start * voxel_size``).
    """
    import mrcfile  # type: ignore[import]

    with mrcfile.open(str(path), mode="r") as mrc:
        shape: tuple[int, int, int] = tuple(mrc.data.shape)  # type: ignore[assignment]
        px = float(mrc.voxel_size.x) or 1.0
        ox = float(mrc.header.origin.x)
        oy = float(mrc.header.origin.y)
        oz = float(mrc.header.origin.z)
        if ox == 0.0 and oy == 0.0 and oz == 0.0:
            ox = float(mrc.header.nxstart) * px
            oy = float(mrc.header.nystart) * px
            oz = float(mrc.header.nzstart) * px
    return shape, px, (ox, oy, oz)


def save_mrc(
    path: str | os.PathLike[str],
    data: torch.Tensor,
    pixel_size: float = 1.0,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Save a float32 tensor as an MRC file.

    Parameters
    ----------
    path : str or Path
        Output path.
    data : torch.Tensor
        ``(d, h, w)`` volume in ZYX order.
    pixel_size : float
        Voxel size in Angstroms.
    origin_xyz : tuple[float, float, float]
        XYZ origin of the map in Angstroms (position of voxel [0,0,0] in world space).
    """
    import mrcfile  # type: ignore[import]
    import numpy as np

    arr = data.cpu().numpy().astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = pixel_size
        mrc.header.origin.x = origin_xyz[0]
        mrc.header.origin.y = origin_xyz[1]
        mrc.header.origin.z = origin_xyz[2]
