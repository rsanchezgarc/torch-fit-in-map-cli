"""Command-line wrapper for :mod:`torch_fit_in_map`.

This package owns all file/format concerns for the ``torch-fit-in-map``
algorithm: reading and writing MRC density maps (``mrcfile``), reading and
writing atomic models as DataFrames (``mmdf``), and the ``typer`` command-line
interfaces.  The algorithm itself lives in ``torch-fit-in-map`` and operates
purely on ``torch.Tensor`` maps and ``pandas.DataFrame`` atom tables.
"""

from importlib.metadata import PackageNotFoundError, version

from ._mrc import load_mrc, read_mrc_header, save_mrc
from ._orchestrate import (
    fit_map_in_map_from_files,
    fit_map_in_pdb_from_files,
    fit_pdb_in_map_from_files,
)
from ._pdb import pdb_centroid_xyz, read_atoms, transform_atomic_model, write_atoms

try:
    __version__ = version("torch-fit-in-map-cli")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "fit_map_in_map_from_files",
    "fit_map_in_pdb_from_files",
    "fit_pdb_in_map_from_files",
    "load_mrc",
    "pdb_centroid_xyz",
    "read_atoms",
    "read_mrc_header",
    "save_mrc",
    "transform_atomic_model",
    "write_atoms",
]
