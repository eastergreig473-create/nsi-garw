"""Check all 35 saved SIFT corruption point estimates with the frozen core."""
from pathlib import Path
import json
import numpy as np
from run_hpatches_suite import DATA, RESULTS, ROOT
from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts, source_sha256


def main():
    destination = RESULTS / "frozen_verification/sift_degradations"
    destination.mkdir(parents=True, exist_ok=True)
    source_hash, rows = source_sha256(), []
    keys = ["A_accessibility", "B_separation", "C_information_purity", "NSI", "geometry_r2", "appearance_r2", "OSR"]
    for kind in ("texture", "illumination", "motion_blur", "jpeg", "repetitive", "occlusion", "composite"):
        for strength in (20, 40, 60, 80, 100):
            identifier = f"sift_degraded_{kind}_s{strength:03d}"
            target = destination / (identifier + ".json")
            features = DATA / "manifests/features" / (identifier + "_clean_attributes.npz")
            fresh = json.loads(target.read_text()) if target.exists() else compute_nsi_artifacts(features, target, load_config(ROOT / "configs/default.toml"))
            original = json.loads((RESULTS / "sift_degradation_nsi" / (identifier + ".json")).read_text())
            first, second = (np.array([record["result"][k] for k in keys]) for record in (fresh, original))
            if not np.allclose(first, second, atol=1e-9, rtol=1e-9):
                raise ValueError(f"Verification mismatch: {identifier}")
            if source_sha256() != source_hash:
                raise ValueError("Core changed during verification")
            rows.append({"condition": identifier, "maximum_absolute_difference": float(np.max(abs(first-second)))})
            print(f"Verified corruption: {identifier}", flush=True)
    (destination / "all_verified.json").write_text(json.dumps({"status": "PASS", "source_sha256": source_hash,
        "condition_count": len(rows), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
