"""Descriptive threshold audit and saved KSG-centering diagnostics; no WSH null test."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from nsi_garw.io import sha256_file
from run_hpatches_suite import MODELS, RESULTS


def main():
    records, inputs = [], {}
    for dataset in ("HPatches", "ETH3D"):
        for identifier, _ in MODELS:
            path = RESULTS / (identifier + ".json") if dataset == "HPatches" else RESULTS / "eth3d" / ("eth3d_" + identifier + ".json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            r = payload["result"]
            records.append({"dataset": dataset, "representation": identifier,
                **{key: r[key] for key in ("NSI", "ksg_observed_mi_nats", "ksg_permutation_median_mi_nats", "ksg_excess_mi_nats", "ksg_sample_count")},
                "null_permutations": payload["provenance"]["config"]["ksg"]["permutations"]})
            inputs[str(path)] = sha256_file(path)
    frame = pd.DataFrame(records)
    thresholds = [.05, .10, .15, .20, .25, .50, .75]
    counts = []
    for dataset, subset in frame.groupby("dataset"):
        for threshold in thresholds:
            counts.append({"dataset": dataset, "threshold": threshold, "strictly_above_count": int((subset.NSI > threshold).sum()), "representation_count": len(subset)})
    out = RESULTS / "threshold_diagnostics"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "pooled_ksg_centering.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "threshold_counts.csv", index=False)
    report = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exploratory descriptive audit after the20 default outcomes; thresholds neither optimized for matching nor interpreted as WSH decisions.",
        "permutation_scope": "Saved200 sample-row appearance-output permutations center the projected KSG estimator for termC. They do not refit probes and are not a scene-level null distribution for full NSI or a formal WSH test. Spatial dependence limits inferential use; no KSG p-value is promoted as a scene-level significance test.",
        "thresholds": thresholds, "counts": counts, "inputs": inputs,
        "script_sha256": sha256_file(Path(__file__)),
        "ranges_by_dataset": {dataset: {column: [float(subset[column].min()), float(subset[column].max())]
            for column in ("NSI", "ksg_observed_mi_nats", "ksg_permutation_median_mi_nats", "ksg_excess_mi_nats")}
            for dataset, subset in frame.groupby("dataset")}}
    (out / "threshold_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(pd.DataFrame(counts).to_string(index=False))
    print(json.dumps(report["ranges_by_dataset"], indent=2))


if __name__ == "__main__":
    main()
