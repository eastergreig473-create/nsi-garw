"""Summarize complete native references; keep their task distinct from GARW retrieval."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nsi_garw.io import sha256_file
from nsi_garw.statistics import benjamini_hochberg, paired_group_bootstrap, spearman_group_association
from run_hpatches_suite import DATA, RESULTS


def main():
    root = RESULTS / "native_matchers"
    summaries, frames, inputs = [], {}, {}
    for model in ("loftr", "roma"):
        for condition, strength in (("clean", "000"), ("texture", "080")):
            folder = root / f"{model}_{condition}_s{strength}"
            source = folder / "matching_summary.json"
            if not source.exists():
                print(f"Incomplete native condition: {folder.name}; no final analysis written", flush=True)
                return
            p = json.loads(source.read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "per_pair.csv")
            if len(frame) != 580 or frame.pair_id.nunique() != 580 or p["processed_pair_count"] != 580:
                raise ValueError(f"Incomplete or duplicated native pairs: {folder}")
            if set(frame.runner_sha256) != {p["runner_sha256"]}:
                raise ValueError("Native runner provenance mismatch")
            if p["pairs_manifest_sha256"] != sha256_file(DATA / "manifests/hpatches/hpatches_pairs.csv"):
                raise ValueError("Native manifest changed")
            test = frame.loc[frame.split == "test"].copy()
            if len(test) != 85 or test.scene.nunique() != 17:
                raise ValueError("Wrong native test partition")
            frames[model, condition] = test
            u = p["results"]["held_out_test_sequences"]
            record = {"model": model, "condition": condition, "test_sequences": 17, "test_pairs": 85,
                "zero_match_pairs": int((test.match_count == 0).sum()),
                "test_mean_match_count": float(test.match_count.mean()),
                "matcher_seconds_median": float(test.matcher_seconds.median()),
                "matcher_seconds_mean": float(test.matcher_seconds.mean()),
                "matcher_seconds_p95": float(test.matcher_seconds.quantile(.95)),
                "cuda_peak_allocated_bytes_max": int(test.cuda_peak_allocated_bytes.max())}
            for threshold in (1, 3, 5, 10):
                metric = u["metrics"][f"mma_at_{threshold}"]
                value = test.groupby("scene")[f"mma_at_{threshold}"].mean().mean()
                if abs(value-metric["estimate"]) > 1e-12:
                    raise ValueError("Saved summary disagrees with per-pair source")
                record.update({f"MMA_{threshold}": metric["estimate"],
                    f"MMA_{threshold}_CI_low": metric["confidence_interval"][0],
                    f"MMA_{threshold}_CI_high": metric["confidence_interval"][1]})
            summaries.append(record)
            inputs[str(source)] = sha256_file(source)
            inputs[str(folder / "per_pair.csv")] = sha256_file(folder / "per_pair.csv")
    comparisons, associations = [], []
    for model in ("loftr", "roma"):
        clean, texture = frames[model, "clean"], frames[model, "texture"]
        joined = texture.merge(clean, on=["pair_id", "scene"], suffixes=("_texture", "_clean"), validate="one_to_one")
        if len(joined) != 85:
            raise ValueError("Native clean/degraded pair mismatch")
        for threshold in (1, 3, 5, 10):
            paired = paired_group_bootstrap(joined[f"mma_at_{threshold}_texture"], joined[f"mma_at_{threshold}_clean"],
                joined.scene, bootstrap_replicates=10000, seed=20261023).to_dict()
            comparisons.append({"model": model, "threshold": threshold, "contrast": "texture0.8 minus clean", **paired})
        identifier = "loftr_outdoor_coarse_1024" if model == "loftr" else "roma_outdoor_scale16_560"
        nsi_source = RESULTS / (identifier + ".json")
        nsi = json.loads(nsi_source.read_text(encoding="utf-8"))
        groups = pd.DataFrame(nsi["groupwise_uncertainty"]["groups"]).rename(columns={"group": "scene"})
        points = groups[["scene", "NSI"]].merge(clean.groupby("scene", as_index=False).mma_at_3.mean(), on="scene", validate="one_to_one")
        if len(points) != 17:
            raise ValueError("NSI/native matching group mismatch")
        points.assign(model=model).to_csv(root / (model + "_nsi_native_MMA3_points.csv"), index=False)
        association = spearman_group_association(points.NSI, points.mma_at_3, points.scene,
            bootstrap_replicates=10000, permutation_replicates=1000, seed=20261023).to_dict()
        associations.append({"model": model, "target": "clean native matcher MMA@3", **association})
        inputs[str(nsi_source)] = sha256_file(nsi_source)
    for row, q in zip(comparisons, benjamini_hochberg([r["p_value_wilcoxon"] for r in comparisons])):
        row["BH_q_8_native_contrasts"] = float(q)
    for row, q in zip(associations, benjamini_hochberg([r["permutation_p_value"] for r in associations])):
        row["BH_q_2_native_associations"] = float(q)
    pd.DataFrame(summaries).to_csv(root / "native_matcher_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(root / "native_clean_texture_paired.csv", index=False)
    pd.DataFrame(associations).to_csv(root / "native_NSI_associations.csv", index=False)
    payload = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "analysis_script_sha256": sha256_file(Path(__file__)),
        "native_runner_sha256": sorted({r for f in frames.values() for r in f.runner_sha256}),
        "accuracy": "17 equal-weight held-out sequences; all 85 pairs including zero-match pairs; RMS symmetric homography error",
        "timing": "3 warmups/model; CUDA-synchronized matcher preprocessing/inference/readout; IO/corruption excluded; peak is total allocated CUDA memory, not incremental GARW overhead; no speed comparison at matched input sizes",
        "comparison_scope": "Native full matcher candidates and match counts differ from encoder top16/Top1; not a controlled comparison against GARW",
        "families": "Exploratory, defined after viewing native clean results:8 within-native texture-minus-clean contrasts (2models x4thresholds) and2 NSI/native-MMA associations. Separate from encoder-GARW16-comparison family and the earlier SIFT association.",
        "summary": summaries, "paired_contrasts": comparisons, "associations": associations}
    (root / "native_analysis_complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries)[["model", "condition", "MMA_3", "matcher_seconds_median", "cuda_peak_allocated_bytes_max"]].to_string(index=False), flush=True)
    print(pd.DataFrame(associations).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
