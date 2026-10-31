"""Scene-level association and ranking analyses of completed real artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nsi_garw.config import load_config
from nsi_garw.experiments import run_association_analysis, run_rank_stability_analysis
from nsi_garw.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/real"


def main():
    config = load_config(ROOT / "configs/default.toml")
    output = RESULTS / "association"
    output.mkdir(parents=True, exist_ok=True)
    nsi_path = RESULTS / "sift_default.json"
    matching_path = RESULTS / "descriptor_matching/sift_default/per_pair.csv"
    nsi = json.loads(nsi_path.read_text(encoding="utf-8"))
    nsi_rows = pd.DataFrame(nsi["groupwise_uncertainty"]["groups"])[["group", "NSI"]].rename(columns={"group": "scene"})
    pairs = pd.read_csv(matching_path)
    pairs = pairs[pairs.split == "test"]
    scene_matching = pairs.groupby("scene", as_index=False)["mma_at_3"].mean()
    table = nsi_rows.merge(scene_matching, on="scene", validate="one_to_one")
    if len(table) != 17:
        raise ValueError("Expected all 17 held-out sequences")
    association_input = output / "sift_clean_sequence_association.csv"
    table.to_csv(association_input, index=False)
    payload = run_association_analysis(association_input, output / "sift_clean_association.json", config,
        x_column="NSI", y_column="mma_at_3", group_column="scene")
    payload["scope"] = "exploratory clean SIFT association across 17 held-out sequences; not cross-model prediction or causal evidence"
    payload["upstream_sha256"] = {str(nsi_path): sha256_file(nsi_path), str(matching_path): sha256_file(matching_path)}
    (output / "sift_clean_association.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=2), flush=True)
    sensitivity = pd.read_csv(RESULTS / "sensitivity/sensitivity_results.csv")
    destination = RESULTS / "sensitivity/ranking_stability.csv"
    rank = run_rank_stability_analysis(RESULTS / "sensitivity/sensitivity_results.csv", destination,
        item_column="experiment_id", variant_column="variant", score_column="NSI", baseline_variant="default")
    print(f"Ranking stability: {len(rank)} alternatives over {sensitivity.experiment_id.nunique()} representations", flush=True)


if __name__ == "__main__":
    main()
