from __future__ import annotations

import unittest

import numpy as np

from nsi_garw.experiments import make_synthetic_garw_table
from nsi_garw.garw import (
    calibrate_garw,
    evaluate_garw,
    garw_scores,
    standardized_projected_cosine,
    split_calibration_evaluation,
)


class GARWTests(unittest.TestCase):
    def test_gate_endpoints_and_conflict_penalty(self) -> None:
        geometry = np.asarray([0.2, 0.9])
        appearance = np.asarray([0.8, 0.1])
        local = np.asarray([0.0, 1.0])
        geometry_only, gate_zero, penalty_zero = garw_scores(
            geometry,
            appearance,
            local,
            alpha=0.0,
            beta=-1000.0,
            penalty_lambda=0.4,
        )
        appearance_only, gate_one, penalty_one = garw_scores(
            geometry,
            appearance,
            local,
            alpha=0.0,
            beta=1000.0,
            penalty_lambda=0.4,
        )
        np.testing.assert_allclose(gate_zero, 0.0, atol=1e-15)
        np.testing.assert_allclose(gate_one, 1.0, atol=1e-15)
        np.testing.assert_allclose(geometry_only, geometry, atol=1e-15)
        np.testing.assert_allclose(appearance_only, appearance, atol=1e-15)
        np.testing.assert_allclose(penalty_zero, 0.0, atol=1e-15)
        np.testing.assert_allclose(penalty_one, 0.0, atol=1e-15)

    def test_standardized_projection_uses_training_statistics(self) -> None:
        first = np.asarray([[10.0, 2.0], [14.0, 3.0]])
        second = np.asarray([[12.0, 2.0], [10.0, 4.0]])
        basis = np.eye(2)
        mean = np.asarray([10.0, 2.0])
        scale = np.asarray([2.0, 1.0])
        score = standardized_projected_cosine(first, second, basis, mean, scale)
        self.assertEqual(score.shape, (2,))
        self.assertTrue(np.all((score >= -1.0) & (score <= 1.0)))

    def test_calibration_and_evaluation(self) -> None:
        table = make_synthetic_garw_table(seed=9)
        calibration, evaluation = split_calibration_evaluation(table)
        parameters, trace = calibrate_garw(
            calibration, [0.0, 0.1], seed=9
        )
        self.assertEqual(len(trace), 2)
        result = evaluate_garw(
            evaluation,
            parameters,
            bootstrap_replicates=100,
            seed=9,
        )
        self.assertIn("garw", result.summary)
        self.assertIn("garw_lambda_zero", result.summary)
        self.assertIn("garw_vs_geometry", result.paired_tests)
        self.assertEqual(result.per_scene["scene"].nunique(), 8)

    def test_scene_leakage_is_rejected(self) -> None:
        table = make_synthetic_garw_table(seed=3)
        evaluation_index = table.index[table["split"] == "evaluation"][0]
        table.loc[evaluation_index, "scene"] = table.loc[table["split"] == "calibration", "scene"].iloc[0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            split_calibration_evaluation(table)

    def test_queries_without_retrieved_positives_count_as_failures(self) -> None:
        table = make_synthetic_garw_table(seed=21)
        calibration, evaluation = split_calibration_evaluation(table)
        parameters, _ = calibrate_garw(calibration, [0.0], seed=21)
        query = evaluation["query_id"].iloc[0]
        evaluation.loc[evaluation["query_id"] == query, "label"] = 0
        result = evaluate_garw(evaluation, parameters, bootstrap_replicates=50, seed=21)
        failed = result.per_query[result.per_query["query_id"] == query]
        self.assertEqual(len(failed), 6)
        np.testing.assert_array_equal(failed["top1"].to_numpy(), 0.)
        np.testing.assert_array_equal(failed["mrr"].to_numpy(), 0.)
        np.testing.assert_array_equal(failed["average_precision"].to_numpy(), 0.)
        self.assertEqual(result.summary["garw"]["scene_count"], 8.)


if __name__ == "__main__":
    unittest.main()
