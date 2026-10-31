"""Geometry checks for matched sparse SfM/laser support, independent of outcomes."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pycolmap

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("annotation_support", SCRIPTS / "run_eth3d_annotation_sensitivity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plane_geometry_and_degenerate_rejection():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    cloud = np.column_stack([x.ravel(), y.ravel(), 4 + .2*x.ravel() - .3*y.ravel()])
    expected = np.asarray([-.2, .3, 1.])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(module.plane_normal(cloud), expected, atol=1e-12)
    assert module.plane_normal(np.column_stack([np.arange(8), np.zeros(8), np.ones(8)])) is None


def test_colmap_half_pixel_and_missing_laser_support():
    camera = pycolmap.Camera(model="PINHOLE", width=8, height=8, params=[10, 10, 4, 4])
    depth = np.full((8, 8), 2.)
    depth[3, 3] = np.nan
    xyz = np.asarray([[.1, .1, 2.], [-.1, -.1, 2.], [0, 0, -1.], [100, 0, 1.]])
    xy, sfm, laser = module.common_clouds(xyz, camera, depth)
    np.testing.assert_allclose(xy, [[4, 4]])
    np.testing.assert_allclose(sfm, [[.1, .1, 2.]])
    np.testing.assert_allclose(laser, sfm)


def test_identical_support_known_tilt_and_radius():
    x, y = np.meshgrid(np.arange(4)*3., np.arange(4)*3.)
    xy = np.column_stack([x.ravel(), y.ravel()])
    first = np.column_stack([x.ravel(), y.ravel(), np.ones(16)*10])
    second = first.copy()
    second[:, 2] += .2*second[:, 0]
    laser, sfm, keep, count = module.paired_normals(np.asarray([[4., 4.], [1000., 1000.]]), xy, first, second)
    np.testing.assert_array_equal(keep, [True, False])
    np.testing.assert_array_equal(count, [16, 0])
    np.testing.assert_allclose(sfm[0], [0, 0, 1.], atol=1e-12)
    expected = np.asarray([-.2, 0, 1.]) / np.sqrt(1.04)
    np.testing.assert_allclose(laser[0], expected, atol=1e-12)
