from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from nsi_garw.config import KSGConfig, PipelineConfig, SplitConfig
from nsi_garw.experiments import make_synthetic_feature_dataset
from nsi_garw.io import FeatureDataset
from nsi_garw.metrics import (
    compute_nsi,
    fit_ridge_probe,
    groupwise_nsi_uncertainty,
    ksg_mutual_information,
    mean_finite,
    normalized_subspace_overlap,
    orthonormal_basis,
    r2_per_output,
    select_ridge_alpha,
)


class SubspaceTests(unittest.TestCase):
    def test_orthogonal_and_identical_overlap(self) -> None:
        first = np.eye(6)[:, :2]
        second = np.eye(6)[:, 2:4]
        self.assertAlmostEqual(normalized_subspace_overlap(first, second), 0.0, places=12)
        self.assertAlmostEqual(normalized_subspace_overlap(first, first), 1.0, places=12)
        mixed = np.column_stack([first[:, 0], (first[:, 1] + second[:, 0]) / np.sqrt(2.0)])
        mean_overlap = normalized_subspace_overlap(first, mixed)
        maximum_overlap = normalized_subspace_overlap(
            first, mixed, metric="maximum_squared_principal_cosine"
        )
        self.assertGreater(maximum_overlap, mean_overlap)

    def test_basis_respects_rank_limit(self) -> None:
        matrix = np.diag([4.0, 2.0, 1.0, 0.0])
        basis = orthonormal_basis(matrix, max_rank=2, relative_tolerance=1e-8)
        self.assertEqual(basis.shape, (4, 2))
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)


class NSITests(unittest.TestCase):
    def test_batched_ksg_counts_match_scalar_reference(self) -> None:
        rng = np.random.default_rng(108)
        x = rng.normal(size=(145, 2))
        y = np.column_stack([x[:, 0] ** 2, rng.normal(size=145)])
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        joint = np.concatenate([x, y], axis=1)
        radii = np.nextafter(cKDTree(joint).query(joint, k=6, p=np.inf)[0][:, -1], 0)
        tx, ty = cKDTree(x), cKDTree(y)
        nx = np.array([len(tx.query_ball_point(a, r, p=np.inf)) - 1 for a, r in zip(x, radii)])
        ny = np.array([len(ty.query_ball_point(a, r, p=np.inf)) - 1 for a, r in zip(y, radii)])
        expected = float(digamma(5) + digamma(len(x)) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
        self.assertAlmostEqual(ksg_mutual_information(x, y, k=5, jitter=0), expected, places=12)

    def _fast_config(self) -> PipelineConfig:
        base = PipelineConfig(seed=123)
        return replace(
            base,
            split=SplitConfig(0.6, 0.2, 0.2),
            ksg=KSGConfig(k=3, permutations=5, max_samples=240, jitter=1e-10),
        )

    def test_nsi_is_bounded_and_separation_detects_entanglement(self) -> None:
        separated = compute_nsi(
            make_synthetic_feature_dataset(
                separated=True, seed=41, group_count=12, samples_per_group=20
            ),
            self._fast_config(),
        )
        entangled = compute_nsi(
            make_synthetic_feature_dataset(
                separated=False, seed=42, group_count=12, samples_per_group=20
            ),
            self._fast_config(),
        )
        for result in (separated, entangled):
            self.assertGreaterEqual(result.nsi, 0.0)
            self.assertLessEqual(result.nsi, 1.0)
            self.assertGreaterEqual(result.osr, 0.0)
            self.assertLessEqual(result.osr, 1.0)
        self.assertGreater(separated.separation, entangled.separation)
        self.assertGreater(separated.nsi, entangled.nsi)

    def test_group_leakage_is_rejected(self) -> None:
        rng = np.random.default_rng(0)
        dataset = FeatureDataset(
            features=rng.normal(size=(12, 5)),
            geometry=rng.normal(size=(12, 2)),
            appearance=rng.normal(size=(12, 1)),
            groups=np.repeat(["a", "b", "c"], 4),
            split=np.asarray(
                ["train", "test", "train", "train"]
                + ["validation"] * 4
                + ["test"] * 4
            ),
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            dataset.validate()

    def test_groupwise_uncertainty_uses_held_out_groups(self) -> None:
        dataset = make_synthetic_feature_dataset(
            separated=True, seed=91, group_count=15, samples_per_group=18
        )
        config = self._fast_config()
        result = compute_nsi(dataset, config)
        uncertainty = groupwise_nsi_uncertainty(dataset, result, config)
        self.assertEqual(uncertainty["group_count"], result.split_group_counts["test"])
        self.assertEqual(uncertainty["global_NSI"], result.nsi)
        self.assertEqual(uncertainty["bootstrap_replicates"], 10000)
        lower, upper = uncertainty["confidence_interval"]
        self.assertLessEqual(lower, upper)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_eigendecomposition_ridge_selection_matches_direct_solves(self) -> None:
        rng = np.random.default_rng(17)
        x_train = rng.normal(size=(80, 12))
        y_train = x_train[:, :3] @ rng.normal(size=(3, 2)) + rng.normal(
            scale=0.1, size=(80, 2)
        )
        x_validation = rng.normal(size=(30, 12))
        y_validation = x_validation[:, :3] @ rng.normal(size=(3, 2))
        lambdas = (1e-6, 1e-2, 1.0, 100.0)
        selected, trace = select_ridge_alpha(
            x_train, y_train, x_validation, y_validation, lambdas
        )
        direct_scores = []
        for alpha in lambdas:
            model = fit_ridge_probe(x_train, y_train, alpha)
            direct_scores.append(
                mean_finite(
                    r2_per_output(y_validation, model.predict(x_validation)),
                    fallback=-np.inf,
                )
            )
        np.testing.assert_allclose(
            [row["validation_r2"] for row in trace], direct_scores, atol=1e-10
        )
        self.assertEqual(selected, lambdas[int(np.argmax(direct_scores))])


if __name__ == "__main__":
    unittest.main()
