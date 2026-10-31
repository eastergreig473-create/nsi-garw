from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nsi_garw.local_proxy import fit_local_nsi_proxy


class LocalProxyTests(unittest.TestCase):
    def test_group_oof_proxy_and_held_out_prediction(self) -> None:
        rng = np.random.default_rng(11)
        rows = []
        for split, start, count in (("calibration", 0, 8), ("evaluation", 8, 3)):
            for scene_index in range(start, start + count):
                for _ in range(10):
                    first, second = rng.uniform(size=2)
                    rows.append(
                        {
                            "scene": f"s{scene_index}",
                            "split": split,
                            "observable_a": first,
                            "observable_b": second,
                            "nsi_reference": np.clip(0.7 * first + 0.2 * second, 0.0, 1.0),
                        }
                    )
        output, model, trace = fit_local_nsi_proxy(
            pd.DataFrame(rows),
            feature_columns=("observable_a", "observable_b"),
            seed=11,
        )
        self.assertTrue(np.isfinite(output["nsi_local"]).all())
        self.assertTrue(np.all((output["nsi_local"] >= 0.0) & (output["nsi_local"] <= 1.0)))
        self.assertEqual(set(output["nsi_mode"]), {"non_oracle_proxy"})
        self.assertLess(model.validation_mse, 0.01)
        self.assertGreater(len(trace), 1)

    def test_proxy_rejects_group_leakage(self) -> None:
        table = pd.DataFrame(
            {
                "scene": ["same", "same"],
                "split": ["calibration", "evaluation"],
                "feature": [0.0, 1.0],
                "nsi_reference": [0.0, 1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            fit_local_nsi_proxy(table, feature_columns=("feature",))


if __name__ == "__main__":
    unittest.main()
