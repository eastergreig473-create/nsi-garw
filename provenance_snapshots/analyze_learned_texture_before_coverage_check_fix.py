"""Consolidate the entire texture grid; never publish a partial model matrix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nsi_garw.io import sha256_file
from run_hpatches_suite import DATA, MODELS, RESULTS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    folder = RESULTS / "learned_texture_nsi"
    expected = [folder / f"{identifier}_texturegrid_s{level:03d}.json"
                for identifier, _ in MODELS if identifier != "sift_default" for level in (20, 40, 60, 80, 100)]
    deadline = time.monotonic()+24*3600
    while not all(p.exists() for p in expected):
        if not args.wait:
            print(f"Incomplete learned grid ({sum(p.exists() for p in expected)}/45); no final summary written", flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Learned texture grid incomplete after24h")
        time.sleep(20)
    inputs, rows = {}, []
    expected_runner = sha256_file(ROOT / "scripts/run_deep_texture_nsi.py")
    for identifier, _ in MODELS:
        for level in (0, 20, 40, 60, 80, 100):
            if not level:
                source = RESULTS / (identifier+".json")
                feature_path = DATA / "manifests/features" / (identifier+".npz")
                build = None
                with np.load(feature_path, allow_pickle=False) as z:
                    metadata = json.loads(str(z["metadata_json"].item()))
                count = 580-len(metadata.get("empty_pairs", []))
            elif identifier == "sift_default":
                source = RESULTS / f"sift_degradation_nsi/sift_degraded_texture_s{level:03d}.json"
                build = source.with_suffix(".build.json")
                feature_path = Path(json.loads(build.read_text(encoding="utf-8"))["output"])
                count = json.loads(build.read_text(encoding="utf-8"))["pairs_with_aligned_features"]
            else:
                source = folder / f"{identifier}_texturegrid_s{level:03d}.json"
                build = source.with_suffix(".build.json")
                report = json.loads(build.read_text(encoding="utf-8"))
                if report["runner_sha256"] != expected_runner or report["pair_count"] != 580:
                    raise ValueError("Learned texture runner/pair provenance mismatch")
                if report["metadata"]["pairs_manifest_sha256"].lower() != sha256_file(DATA / "manifests/hpatches/hpatches_pairs.csv"):
                    raise ValueError("Learned texture pairs manifest changed")
                feature_path, count = Path(report["output"]), report["pairs_with_aligned_features"]
            payload = json.loads(source.read_text(encoding="utf-8"))
            actual_sha = sha256_file(feature_path)
            if actual_sha != payload["provenance"]["input_sha256"].lower():
                raise ValueError("Texture result input export changed")
            if build is not None:
                b = json.loads(build.read_text(encoding="utf-8"))
                if actual_sha != b["output_sha256"].lower():
                    raise ValueError("Texture build/result export SHA differs")
                inputs[str(build)] = sha256_file(build)
            result, u = payload["result"], payload["groupwise_uncertainty"]
            if result["split_group_counts"] != {"train": 81, "validation": 18, "test": 17}:
                raise ValueError("A texture condition lost a frozen scene group")
            row = {"representation": identifier, "strength": level/100,
                   **{k: result[k] for k in ("A_accessibility", "B_separation", "C_information_purity", "NSI", "geometry_r2", "appearance_r2")},
                   "sequence_NSI_mean": u["group_balanced_NSI_mean"],
                   "sequence_NSI_CI_low": u["confidence_interval"][0], "sequence_NSI_CI_high": u["confidence_interval"][1],
                   "aligned_pairs": count, "sample_count": sum(result["split_counts"].values()),
                   "test_sequences": u["group_count"], "result_json": str(source)}
            rows.append(row)
            inputs[str(source)] = sha256_file(source)
            inputs[str(feature_path)] = actual_sha
    frame = pd.DataFrame(rows)
    if len(frame) != 60 or frame.duplicated(["representation", "strength"]).any():
        raise ValueError("Texture matrix is not10representations x6strengths")
    curves = []
    for identifier, block in frame.groupby("representation", sort=False):
        block = block.sort_values("strength")
        steps = np.diff(block.NSI.to_numpy())
        curves.append({"representation": identifier, "clean_NSI": float(block.NSI.iloc[0]),
                       "endpoint_NSI": float(block.NSI.iloc[-1]),
                       "minimum_aligned_pairs": int(block.aligned_pairs.min()),
                       "positive_steps": int((steps>1e-9).sum()), "negative_steps": int((steps < -1e-9).sum()),
                       "non_increasing_pooled_NSI": bool((steps<=1e-9).all())})
    frame.to_csv(folder / "texture_grid_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(folder / "texture_curve_diagnostics.csv", index=False)
    manifest = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha256_file(Path(__file__)), "learned_runner_sha256": expected_runner,
                "inputs": inputs, "rows": len(rows), "learned_conditions": 45, "curves": curves,
                "scope": "Exploratory; fixed81/18/17sequences, clean scalar NCC/Sobel targets, strength-specific probe refits; conditional surviving-native-feature coverage. No fixed-keypoint trajectories, matching accuracy, universal decay law or collapse-rate fit inferred from NSI curves.",
                "uncertainty": "10000sequence bootstraps with condition-specific frozen fitted probes; excludes probe-refit uncertainty; not a cross-condition simultaneous band"}
    (folder / "texture_analysis_complete.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(pd.DataFrame(curves).to_string(index=False),flush=True)
    print("Texture matrix complete:60rows,45learned corruptions,all export SHAs verified",flush=True)


if __name__ == "__main__":
    main()
