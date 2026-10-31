from pathlib import Path
import numpy as np
import pytest

from nsi_garw.attributes import depth_to_normals
from nsi_garw.eth3d import depth_on_pinhole, read_raw_depth


def test_depth_stencil_rejects_missing_neighbors():
    depth = np.ones((7, 7))
    depth[3, 3] = np.inf
    normals, valid = depth_to_normals(depth, (10., 10., 3., 3.))
    for y, x in [(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)]:
        assert not valid[y, x]
        np.testing.assert_array_equal(normals[y, x], 0.)
    assert valid[1, 1]


def test_raw_depth_length_and_identity_mapping(tmp_path: Path):
    pycolmap = pytest.importorskip("pycolmap")
    source = np.arange(1, 31, dtype="<f4").reshape(5, 6)
    source[2, 3] = np.inf
    path = tmp_path / "not_actually_a_jpeg.JPG"
    source.tofile(path)
    depth = read_raw_depth(path, 6, 5)
    camera = pycolmap.Camera(model="PINHOLE", width=6, height=5, params=[10., 10., 3., 2.5])
    actual = depth_on_pinhole(depth, camera, camera, block_rows=2)
    np.testing.assert_allclose(actual, depth, equal_nan=True)
    with pytest.raises(ValueError, match="size disagrees"):
        read_raw_depth(path, 7, 5)
    narrow_original = pycolmap.Camera(model="PINHOLE", width=6, height=5, params=[100., 100., 3., 2.5])
    mapped = depth_on_pinhole(depth, narrow_original, camera)
    assert np.isnan(mapped[0, 0])
    assert np.isnan(mapped[-1, -1])
