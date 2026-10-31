"""Follow completed SIFT stress matching with separately refitted, held-out NSI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from nsi_garw.config import load_config
from nsi_garw.degradations import DEGRADATIONS
from nsi_garw.experiments import compute_nsi_artifacts
from nsi_garw.feature_builders import build_hpatches_feature_dataset
from run_hpatches_suite import DATA, RESULTS


class CachedOnly:
    def extract(self, image):
        raise RuntimeError("Completed condition is missing a feature cache; rebuild matching first")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    config = load_config(Path("configs/default.toml"))
    for kind in DEGRADATIONS:
        for strength in [.2, .4, .6, .8, 1.]:
            identifier = f"sift_degraded_{kind}_s{round(strength * 100):03d}"
            matching = RESULTS / "sift_degradation_matching" / identifier / "matching_summary.json"
            destination = RESULTS / "sift_degradation_nsi" / f"{identifier}.json"
            if destination.exists():
                continue
            deadline = time.monotonic() + 8 * 3600
            while not matching.exists() and args.wait and time.monotonic() < deadline:
                time.sleep(20)
            if not matching.exists():
                print(f"Pending matching: {identifier}", flush=True)
                continue
            output = DATA / "manifests/features" / (identifier + "_clean_attributes.npz")
            report = build_hpatches_feature_dataset(CachedOnly(),
                DATA / "manifests/hpatches/hpatches_pairs.csv",
                DATA / "datasets/HPatches/hpatches-sequences-release", output, DATA / "cache",
                export_id=identifier, max_samples_per_pair=64, seed=config.seed,
                expected_pairs=580, proxy="ncc", allow_empty_pairs=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            provenance = destination.with_suffix(".build.json")
            report.update({"protocol": "refit probes separately per corruption condition using unchanged 81/18/17 sequence split",
                "attribute_targets": "clean-image homography NCC and Sobel attributes sampled at degraded-image detected keypoints",
                "interpretation": "conditional representation accessibility on surviving features; detector coverage reported separately",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            provenance.write_text(json.dumps(report, indent=2), encoding="utf-8")
            payload = compute_nsi_artifacts(output, destination, config)
            print(f"DONE {identifier}: NSI={payload['result']['NSI']:.6f}, "
                  f"coverage={report['pairs_with_aligned_features']}/580", flush=True)


if __name__ == "__main__":
    main()
