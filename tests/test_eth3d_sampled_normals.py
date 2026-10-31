import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("eth3d_suite", Path(__file__).resolve().parents[1] / "scripts/run_eth3d_suite.py")
module = importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec.loader.exec_module(module)


def test_perspective_slanted_plane_and_colmap_pixel_centers():
    y, x = np.indices((11, 13), dtype=float)
    fx, fy, cx, cy = 20., 18., 6.5, 5.5
    a, b, distance = .3, -.2, 4.
    depth = distance / (1 - a*(x+.5-cx)/fx - b*(y+.5-cy)/fy)
    coordinates = np.array([[3., 3.], [6.2, 5.3], [9., 7.]])
    normals, valid = module.sampled_normals(depth, coordinates, (fx, fy, cx, cy))
    expected = np.array([-a, -b, 1.])
    expected /= np.linalg.norm(expected)
    assert valid.all()
    np.testing.assert_allclose(normals, np.tile(expected, (3, 1)), atol=1e-12)


def test_missing_neighbor_and_outside_points_are_rejected():
    depth = np.full((7, 7), 3.)
    depth[3, 2] = np.nan
    coordinates = np.array([[3., 3.], [4., 4.], [0., 2.], [-1., 2.], [np.nan, 3.]])
    normals, valid = module.sampled_normals(depth, coordinates, (10., 10., 3.5, 3.5))
    assert valid.tolist() == [False, True, False, False, False]
    np.testing.assert_allclose(normals[1], [0., 0., 1.])
    np.testing.assert_array_equal(normals[~valid], 0.)
