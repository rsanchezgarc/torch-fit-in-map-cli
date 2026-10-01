"""Tests for MRC reading/writing helpers."""

import numpy as np
import torch


def test_save_load_roundtrip(tmp_path):
    from torch_fit_in_map_cli import load_mrc, save_mrc

    data = torch.rand(12, 10, 8)
    path = tmp_path / "vol.mrc"
    save_mrc(path, data, pixel_size=1.7, origin_xyz=(3.0, 4.0, 5.0))

    loaded, px = load_mrc(path)
    assert loaded.shape == (12, 10, 8)
    assert abs(px - 1.7) < 1e-4
    np.testing.assert_allclose(loaded.numpy(), data.numpy(), atol=1e-4)


def test_read_mrc_header_origin(tmp_path):
    from torch_fit_in_map_cli import read_mrc_header, save_mrc

    data = torch.zeros(6, 6, 6)
    path = tmp_path / "vol.mrc"
    save_mrc(path, data, pixel_size=2.0, origin_xyz=(1.0, 2.0, 3.0))

    shape, px, origin = read_mrc_header(path)
    assert shape == (6, 6, 6)
    assert abs(px - 2.0) < 1e-4
    np.testing.assert_allclose(origin, (1.0, 2.0, 3.0), atol=1e-4)


def test_start_origin_matches_explicit_origin(tmp_path):
    import mrcfile

    from torch_fit_in_map_cli import read_mrc_header, save_mrc

    path = tmp_path / "start.mrc"
    save_mrc(path, torch.zeros(7, 8, 9), pixel_size=1.5)
    with mrcfile.open(path, mode="r+") as mrc:
        mrc.header.nxstart = -3
        mrc.header.nystart = 4
        mrc.header.nzstart = 7
    shape, px, origin = read_mrc_header(path)
    assert shape == (7, 8, 9)
    assert px == 1.5
    assert origin == (-4.5, 6.0, 10.5)
    # An explicit nonzero origin takes precedence over legacy starts.
    with mrcfile.open(path, mode="r+") as mrc:
        mrc.header.origin.x = 2
    assert read_mrc_header(path)[2] == (2.0, 0.0, 0.0)
