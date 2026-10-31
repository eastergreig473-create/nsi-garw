"""Add paired comparisons to completed fixed per-query outputs."""
import json
from pathlib import Path
import pandas as pd
from nsi_garw.statistics import paired_group_bootstrap
from nsi_garw.io import sha256_file
from nsi_garw.experiments import source_sha256

ROOT = Path(__file__).resolve().parents[1] / "results/real/garw_hpatches"
for path in ROOT.rglob("garw_summary.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_query_path = path.with_name("garw_per_query.csv")
    table = pd.read_csv(per_query_path)
    scenes = table.drop_duplicates("query_id").set_index("query_id").scene
    changed = False
    for metric in ("average_precision", "top1"):
        pivot = table.pivot(index="query_id", columns="method", values=metric)
        for baseline in ("descriptor_baseline", "fixed_weight"):
            key = f"garw_vs_{baseline}_{metric}"
            if key in payload["paired_tests"] or baseline not in pivot:
                continue
            result = paired_group_bootstrap(pivot.garw.to_numpy(), pivot[baseline].to_numpy(),
                scenes.loc[pivot.index].to_numpy(), bootstrap_replicates=10000, confidence_level=.95, seed=20261026)
            payload["paired_tests"][key] = result.to_dict()
            changed = True
    if changed:
        payload["provenance"]["additional_paired_comparisons"] = {
            "fixed_per_query_sha256": sha256_file(per_query_path), "source_sha256": source_sha256(),
            "calibration": "unchanged; comparisons operate only on saved fixed predictions"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Added paired baselines: {path.parent}", flush=True)
