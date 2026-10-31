"""Recheck the completed cohort against the now-stable metric source tree."""
from pathlib import Path
import csv
import json
import numpy as np
from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts, source_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/real/frozen_verification"


def main():
    config = load_config(ROOT / "configs/default.toml")
    source = source_sha256()
    with (ROOT / "manifests/hpatches_nsi_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for record in manifest:
        identifier = record["experiment_id"]
        destination = OUTPUT / (identifier + ".json")
        if destination.exists():
            fresh = json.loads(destination.read_text(encoding="utf-8"))
        else:
            fresh = compute_nsi_artifacts(record["input_npz"], destination, config)
        original = json.loads((ROOT / "results/real" / (identifier + ".json")).read_text(encoding="utf-8"))
        keys = ["A_accessibility", "B_separation", "C_information_purity", "NSI", "geometry_r2", "appearance_r2", "OSR"]
        first = np.asarray([original["result"][key] for key in keys])
        second = np.asarray([fresh["result"][key] for key in keys])
        if not np.allclose(first, second, atol=1e-9, rtol=1e-9):
            raise ValueError(f"Stable-source verification mismatch: {identifier}: {first} vs {second}")
        if source_sha256() != source:
            raise RuntimeError("Metric source tree changed during cohort verification")
        reports.append({"experiment_id": identifier, "max_absolute_difference": float(np.max(np.abs(first-second))), "passed": True})
        (OUTPUT / "verification_progress.json").write_text(json.dumps({"source_sha256": source, "rows": reports}, indent=2), encoding="utf-8")
        print(f"Verified stable-source NSI: {identifier}", flush=True)
    (OUTPUT / "all_verified.json").write_text(json.dumps({"source_sha256": source,
        "representation_count": len(reports), "rows": reports, "status": "PASS"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
