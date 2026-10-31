from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nsi_garw.evaluation import (
    depth_pose_reprojection_errors,
    evaluate_match_table,
    homography_reprojection_errors,
)


class EvaluationTests(unittest.TestCase):
    def test_homography_error(self) -> None:
        source = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        homography = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
        destination = source + np.asarray([2.0, -1.0])
        errors = homography_reprojection_errors(source, destination, homography, symmetric=True)
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_depth_pose_error(self) -> None:
        points = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        depth = np.asarray([2.0, 3.0])
        errors = depth_pose_reprojection_errors(
            points,
            depth,
            points,
            np.eye(3),
            np.eye(3),
            np.eye(4),
        )
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_match_table_and_paired_inference(self) -> None:
        rows = []
        for scene_index in range(4):
            for pair_index in range(2):
                pair = f"s{scene_index}_p{pair_index}"
                for method, error in (("base", 6.0), ("GARW", 2.0)):
                    for _ in range(5):
                        rows.append(
                            {
                                "pair_id": pair,
                                "scene": f"s{scene_index}",
                                "method": method,
                                "reprojection_error": error,
                            }
                        )
        result = evaluate_match_table(
            pd.DataFrame(rows),
            thresholds=(3.0, 10.0),
            primary_method="GARW",
            baseline_methods=("base",),
            bootstrap_replicates=100,
            seed=5,
        )
        self.assertAlmostEqual(result.summary["GARW"]["mma_at_3"], 1.0)
        self.assertAlmostEqual(result.summary["base"]["mma_at_3"], 0.0)
        self.assertIn("GARW_vs_base_mma_at_10", result.paired_tests)


if __name__ == "__main__":
    unittest.main()
