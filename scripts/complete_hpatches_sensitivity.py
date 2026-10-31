"""Resume the full cross-model sensitivity grid with explicit artifact reuse."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from nsi_garw.config import load_config
from nsi_garw.experiments import run_sensitivity, source_sha256, config_sha256
from nsi_garw.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/real/sensitivity"


def main():
    config = load_config(ROOT / "configs/default.toml")
    manifest_path = ROOT / "manifests/hpatches_nsi_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    frozen_source = source_sha256()
    reused, generated, tables = [], [], []
    for record in manifest.to_dict(orient="records"):
        source = Path(record["input_npz"])
        digest = sha256_file(source)
        destination = OUTPUT / "by_experiment" / (record["experiment_id"] + ".csv")
        if destination.exists():
            table = pd.read_csv(destination)
            if set(table.input_sha256) != {digest} or set(table.experiment_id) != {record["experiment_id"]}:
                raise ValueError(f"Cached sensitivity artifact does not match its input: {destination}")
            baseline = table[table.variant == "default"]
            if len(baseline) != 1 or baseline.config_sha256.iloc[0] != config_sha256(config):
                raise ValueError(f"Cached sensitivity baseline configuration changed: {destination}")
            reused.append({"file": str(destination), "sha256": sha256_file(destination)})
            print(f"Verified existing sensitivity: {record['experiment_id']}", flush=True)
        else:
            print(f"Sensitivity START: {record['experiment_id']}", flush=True)
            table = run_sensitivity(source, destination, config)
            for column in ("experiment_id", "model", "dataset", "annotation_source"):
                table.insert(0, column, record[column])
            table["input_npz"], table["input_sha256"] = str(source), digest
            table.to_csv(destination, index=False)
            generated.append({"file": str(destination), "sha256": sha256_file(destination)})
            print(f"Sensitivity DONE: {record['experiment_id']}", flush=True)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(OUTPUT / "sensitivity_results.csv", index=False)
    (OUTPUT / "completion_provenance.json").write_text(json.dumps({"source_sha256_start": frozen_source,
        "source_sha256_end": source_sha256(), "config_sha256": config_sha256(config),
        "manifest_sha256": sha256_file(manifest_path), "reused_artifacts": reused,
        "new_artifacts": generated, "scope": "full completed HPatches cross-model manifest"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
