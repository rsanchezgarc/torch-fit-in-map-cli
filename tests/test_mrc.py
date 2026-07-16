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
