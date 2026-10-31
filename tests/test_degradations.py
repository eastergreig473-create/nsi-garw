from __future__ import annotations

import unittest

import numpy as np

from nsi_garw.degradations import DEGRADATIONS, apply_degradation, jpeg_compression


class DegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        rows, columns = np.indices((48, 64))
        self.image = np.stack(
            [rows / 47.0, columns / 63.0, ((rows + columns) % 11) / 10.0],
            axis=-1,
        )

    def test_zero_strength_is_identity(self) -> None:
        for name in DEGRADATIONS:
            with self.subTest(name=name):
                output = apply_degradation(self.image, name, 0.0, seed=7)
                np.testing.assert_allclose(output, self.image, atol=0.0)

    def test_outputs_are_deterministic_and_bounded(self) -> None:
        for name in DEGRADATIONS:
            with self.subTest(name=name):
                first = apply_degradation(self.image, name, 0.7, seed=19)
                second = apply_degradation(self.image, name, 0.7, seed=19)
                self.assertEqual(first.shape, self.image.shape)
                np.testing.assert_allclose(first, second, atol=0.0)
                self.assertTrue(np.isfinite(first).all())
                self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))

    def test_invalid_strength_fails(self) -> None:
        with self.assertRaises(ValueError):
            apply_degradation(self.image, "jpeg", 1.1)

    def test_jpeg_preserves_gray_and_auxiliary_channel_shapes(self) -> None:
        gray = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
        one_channel = gray[..., None]
        two_channel = np.stack([gray, 1.0 - gray], axis=-1)
        rgba = np.concatenate(
            [np.repeat(one_channel, 3, axis=-1), np.full_like(one_channel, 0.4)],
            axis=-1,
        )
        for image in (gray, one_channel, two_channel, rgba):
            with self.subTest(shape=image.shape):
                degraded = jpeg_compression(image, 0.6)
                self.assertEqual(degraded.shape, image.shape)
                self.assertTrue(np.isfinite(degraded).all())


if __name__ == "__main__":
    unittest.main()
