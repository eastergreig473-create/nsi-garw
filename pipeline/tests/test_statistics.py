from __future__ import annotations

import unittest

import numpy as np

from nsi_garw.statistics import (
    benjamini_hochberg,
    paired_group_bootstrap,
    ranking_stability,
    spearman_group_association,
)


class StatisticsTests(unittest.TestCase):
    def test_paired_group_bootstrap(self) -> None:
        groups = np.repeat([f"s{index}" for index in range(8)], 5)
        second = np.linspace(0.1, 0.6, groups.size)
        first = second + 0.12
        result = paired_group_bootstrap(
            first, second, groups, bootstrap_replicates=300, seed=4
        )
        self.assertAlmostEqual(result.estimate, 0.12, places=12)
        self.assertGreater(result.confidence_interval[0], 0.0)

    def test_spearman_and_rank_stability(self) -> None:
        x = np.arange(12, dtype=np.float64)
        y = 2.0 * x + 1.0
        association = spearman_group_association(
            x,
            y,
            bootstrap_replicates=100,
            permutation_replicates=100,
            seed=2,
        )
        self.assertAlmostEqual(association.spearman_rho, 1.0, places=12)
        stability = ranking_stability(x, {"same": x, "reverse": -x})
        self.assertAlmostEqual(stability[0].kendall_tau, 1.0, places=12)
        self.assertAlmostEqual(stability[1].spearman_rho, -1.0, places=12)

    def test_benjamini_hochberg_is_monotonic_in_rank(self) -> None:
        p = np.asarray([0.01, 0.04, 0.03, 0.20])
        adjusted = benjamini_hochberg(p)
        self.assertTrue(np.all((adjusted >= p) & (adjusted <= 1.0)))
        ordered = adjusted[np.argsort(p)]
        self.assertTrue(np.all(np.diff(ordered) >= -1e-15))


if __name__ == "__main__":
    unittest.main()
