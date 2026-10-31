"""Summarize completed cross-dataset, corruption, and gate experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_hpatches_suite import ROOT, RESULTS, MODELS
from nsi_garw.io import sha256_file
from nsi_garw.statistics import benjamini_hochberg

CONDITIONS = {
    "LoFTR clean": "loftr_outdoor_coarse_1024",
    "RoMa clean": "roma_outdoor_scale16_560_cuda_amp",
    "LoFTR texture 0.8": "loftr_outdoor_coarse_1024_texture_s080",
    "RoMa texture 0.8": "roma_outdoor_scale16_560_texture_s080_cuda_amp",
}


def main():
    eth_rows = []
    for identifier, _ in MODELS:
        source = RESULTS / "eth3d" / ("eth3d_" + identifier + ".json")
        build = source.with_suffix(".build.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
        report = json.loads(build.read_text(encoding="utf-8"))
        u, r = payload["groupwise_uncertainty"], payload["result"]
        eth_rows.append({"experiment_id": identifier, "model": report["metadata"]["model"],
            "dataset": "ETH3D", "samples": report["sample_count"], "dimension": report["feature_dimension"],
            **{k: r[k] for k in ("A_accessibility", "B_separation", "C_information_purity", "NSI", "geometry_r2", "appearance_r2", "OSR")},
            "group_NSI_mean": u["group_balanced_NSI_mean"], "group_NSI_CI_low": u["confidence_interval"][0],
            "group_NSI_CI_high": u["confidence_interval"][1], "test_groups": u["group_count"],
            "input_images": len(report["image_coverage"]), "sampled_images": sum(v["samples"] > 0 for v in report["image_coverage"]),
            "empty_images": sum(v["samples"] == 0 for v in report["image_coverage"]),
            "result_json": str(source), "result_sha256": sha256_file(source), "build_sha256": sha256_file(build)})
    pd.DataFrame(eth_rows).to_csv(RESULTS / "eth3d_nsi_summary.csv", index=False)
    hp = pd.read_csv(RESULTS / "hpatches_nsi_summary.csv")
    pd.concat([hp.assign(dataset="HPatches"), pd.DataFrame(eth_rows)], ignore_index=True).to_csv(RESULTS / "cross_dataset_nsi_summary.csv", index=False)

    stress_rows = []
    for kind in ("texture", "illumination", "motion_blur", "jpeg", "repetitive", "occlusion", "composite"):
        for strength in (.0, .2, .4, .6, .8, 1.):
            identifier = f"sift_degraded_{kind}_s{round(strength*100):03d}"
            source = RESULTS / ("sift_default.json" if strength == 0 else f"sift_degradation_nsi/{identifier}.json")
            matching = RESULTS / ("descriptor_matching/sift_default/matching_summary.json" if strength == 0
                else f"sift_degradation_matching/{identifier}/matching_summary.json")
            nsi = json.loads(source.read_text(encoding="utf-8"))
            match = json.loads(matching.read_text(encoding="utf-8"))
            u, r = nsi["groupwise_uncertainty"], nsi["result"]
            metric = match["results"]["held_out_test_sequences"]["metrics"]["mma_at_3"]
            coverage = 580 if strength == 0 else json.loads(source.with_suffix(".build.json").read_text())["pairs_with_aligned_features"]
            stress_rows.append({"condition": kind, "strength": strength, "NSI": r["NSI"],
                "sequence_NSI_mean": u["group_balanced_NSI_mean"], "NSI_ci_low": u["confidence_interval"][0],
                "NSI_ci_high": u["confidence_interval"][1], "MMA_at_3_test": metric["estimate"],
                "MMA_ci_low": metric["confidence_interval"][0], "MMA_ci_high": metric["confidence_interval"][1],
                "MMA_at_3_all_descriptive": match["results"]["all_sequences_descriptive"]["metrics"]["mma_at_3"]["estimate"],
                "aligned_pairs": coverage, "all_pairs": 580, "nsi_json": str(source), "matching_json": str(matching),
                "nsi_sha256": sha256_file(source), "matching_sha256": sha256_file(matching)})
    pd.DataFrame(stress_rows).to_csv(RESULTS / "sift_degradation_summary.csv", index=False)

    groups = json.loads((RESULTS / "sift_default.json").read_text())["groupwise_uncertainty"]["groups"]
    strata = []
    rng = np.random.default_rng(20261023)
    for name, prefix in (("illumination", "i_"), ("viewpoint", "v_")):
        values = np.asarray([v["NSI"] for v in groups if v["group"].startswith(prefix)])
        boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        strata.append({"stratum": name, "sequence_count": len(values), "sequence_NSI_mean": float(values.mean()),
                       "confidence_interval": np.quantile(boot, [.025, .975]).tolist()})
    (RESULTS / "hpatches_natural_strata.json").write_text(json.dumps({"strata": strata,
        "scope": "descriptive split of the same 17 held-out SIFT sequences; different scenes; not an isolated illumination intervention",
        "bootstrap_replicates": 10000}, indent=2), encoding="utf-8")

    comparisons, inputs = [], {}
    for condition, identifier in CONDITIONS.items():
        source = RESULTS / "garw_hpatches" / identifier / "non_oracle/garw_summary.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        inputs[str(source)] = sha256_file(source)
        for metric in ("top1", "average_precision"):
            for baseline in ("descriptor_baseline", "fixed_weight"):
                key = f"garw_vs_{baseline}_{metric}"
                row = payload["paired_tests"][key]
                comparisons.append({"condition": condition, "metric": metric, "baseline": baseline,
                    "difference": row["estimate"], "ci_low": row["confidence_interval"][0], "ci_high": row["confidence_interval"][1],
                    "wilcoxon_p": row["p_value_wilcoxon"], "summary_json": str(source)})
    adjusted = benjamini_hochberg([v["wilcoxon_p"] for v in comparisons])
    for row, q in zip(comparisons, adjusted):
        row["BH_q_16_comparisons"] = float(q)
    pd.DataFrame(comparisons).to_csv(RESULTS / "garw_hpatches/exploratory_multiple_comparisons.csv", index=False)
    (RESULTS / "revision_cohort_provenance.json").write_text(json.dumps({"gate_inputs": inputs,
        "gate_family": "16 exploratory comparisons: four conditions x two metrics x two baselines; family defined after outcomes were examined; no confirmatory preregistration claim",
        "corruption_conditions": 35, "eth3d_representations": len(eth_rows), "script_sha256": sha256_file(Path(__file__))}, indent=2), encoding="utf-8")
    print(pd.DataFrame(eth_rows)[["model", "NSI", "sampled_images", "empty_images"]].to_string(index=False))
    print(pd.DataFrame(comparisons).query("metric == 'top1' and baseline == 'descriptor_baseline'").to_string(index=False))


if __name__ == "__main__":
    main()
