from __future__ import annotations

import unittest

import numpy as np

from nsi_garw.attributes import (
    depth_to_normals,
    sobel_magnitude,
    structure_tensor,
    tensor_cosine_proxy,
    tensor_ncc_proxy,
    sample_map_at_coordinates,
    warp_structure_tensor,
)


class AttributeTests(unittest.TestCase):
    def test_planar_depth_normals(self) -> None:
        _, columns = np.indices((20, 24), dtype=np.float64)
        depth = 1.0 + 0.1 * columns
        normals, valid = depth_to_normals(depth)
        expected = np.asarray([-0.1, 0.0, 1.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normals[5:-5, 5:-5].mean(axis=(0, 1)), expected, atol=1e-8)
        self.assertTrue(valid.all())

    def test_identity_homography_preserves_tensor(self) -> None:
        image = np.zeros((32, 40), dtype=np.float64)
        image[:, 20:] = 1.0
        tensor = structure_tensor(image)
        warped, valid = warp_structure_tensor(tensor, np.eye(3), image.shape)
        np.testing.assert_allclose(warped, tensor, atol=1e-10)
        self.assertTrue(valid.all())
        proxy = tensor_cosine_proxy(tensor, warped, valid)
        energetic = np.linalg.norm(tensor, axis=-1) > 1e-10
        np.testing.assert_allclose(proxy[energetic], 1.0, atol=1e-10)
        ncc = tensor_ncc_proxy(tensor, warped, valid, sigma=1.5)
        informative = np.isfinite(ncc) & (np.abs(ncc) > 1e-8)
        np.testing.assert_allclose(ncc[informative], 1.0, atol=1e-9)

    def test_sobel_is_bounded(self) -> None:
        image = np.tile(np.linspace(0.0, 1.0, 32), (20, 1))
        magnitude = sobel_magnitude(image)
        self.assertEqual(magnitude.shape, image.shape)
        self.assertTrue(np.all((magnitude >= 0.0) & (magnitude <= 1.0)))

    def test_coordinate_sampling(self) -> None:
        value = np.arange(20, dtype=np.float64).reshape(4, 5)
        sampled, valid = sample_map_at_coordinates(
            value, np.asarray([[0.0, 0.0], [4.0, 3.0], [7.0, 1.0]])
        )
        np.testing.assert_allclose(sampled[:2, 0], [0.0, 19.0])
        self.assertEqual(valid.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
