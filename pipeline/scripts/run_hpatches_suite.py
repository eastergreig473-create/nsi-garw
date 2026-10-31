"""Resumable real HPatches feature/NSI runner; all large outputs stay on F:."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("F:/NSI_GARW_Data")
FEATURES = DATA / "manifests/features"
RESULTS = ROOT / "results/real"
LOGS = DATA / "logs/hpatches_suite"

# Identifiers declare feature type and preprocessing.  No model is retrained.
MODELS = [
    ("sift_default", ["--adapter", "sift", "--max-keypoints", "4096"]),
    ("xfeat_e92685f_top4096_thr005", ["--adapter", "xfeat", "--repository", "external/xfeat", "--checkpoint", str(DATA / "models/xfeat.pt"), "--max-keypoints", "4096", "--threshold", "0.05"]),
    ("superpoint_eb42fee_top4096_resize1024", ["--adapter", "superpoint", "--repository", "external/lightglue", "--checkpoint", "weights/superpoint_v1.pth", "--max-keypoints", "4096", "--image-size", "1024"]),
    ("dinov2_vits14_block11_518", ["--adapter", "dinov2", "--repository", "external/dinov2", "--checkpoint", str(DATA / "models/dinov2_vits14_pretrain.pth"), "--layer", "11", "--image-size", "518"]),
    ("dedode_l_v2_b_top4096_resize784", ["--adapter", "dedode", "--repository", "external/dedode", "--checkpoint", str(DATA / "models/dedode_detector_L_v2.pth"), "--aux-checkpoint", str(DATA / "models/dedode_descriptor_B.pth"), "--max-keypoints", "4096", "--image-size", "784"]),
    ("loftr_outdoor_coarse_1024", ["--adapter", "loftr", "--checkpoint", str(DATA / "models/loftr_outdoor.ckpt"), "--layer", "coarse", "--image-size", "1024"]),
    ("roma_outdoor_scale16_560", ["--adapter", "roma", "--repository", "external/roma", "--checkpoint", str(DATA / "models/roma_outdoor.pth"), "--aux-checkpoint", str(DATA / "models/dinov2_vitl14_pretrain.pth"), "--layer", "16", "--image-size", "560"]),
    ("vggt_1b_layer23_518", ["--adapter", "vggt", "--repository", "external/vggt", "--checkpoint", str(DATA / "models/vggt_1b_model.pt"), "--layer", "23", "--image-size", "518"]),
    ("mast3r_encoder23_512", ["--adapter", "mast3r", "--repository", "external/mast3r", "--checkpoint", str(DATA / "models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"), "--layer", "23", "--image-size", "512"]),
    ("dust3r_encoder23_512", ["--adapter", "dust3r", "--repository", "external/mast3r", "--checkpoint", str(DATA / "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"), "--layer", "23", "--image-size", "512"]),
]


def run_command(identifier: str, phase: str, arguments: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS / f"{identifier}_{phase}_{stamp}.log"
    command = [sys.executable, "-m", "nsi_garw.cli", *arguments]
    print(json.dumps({"status": "started", "id": identifier, "phase": phase,
                      "log": str(log_path), "command": command}), flush=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update({"TORCH_HOME": str(DATA / "cache/torch"),
                        "HF_HOME": str(DATA / "cache/huggingface"),
                        "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "4",
                        "OMP_NUM_THREADS": "4"})
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if completed.returncode:
        raise RuntimeError(f"{identifier} {phase} failed ({completed.returncode}); see {log_path}")
    print(json.dumps({"status": "completed", "id": identifier, "phase": phase,
                      "wall_seconds": time.perf_counter() - started}), flush=True)


def summarize() -> None:
    metadata_keys = ["model", "dataset", "layer", "implementation", "implementation_commit",
                     "checkpoint", "checkpoint_sha256", "preprocessing", "coordinate_system",
                     "annotation_source", "split_id"]
    manifest_rows, result_rows = [], []
    for identifier, _ in MODELS:
        path = FEATURES / f"{identifier}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"].item()))
            sample_count, dimension = z["features"].shape
        record = {"experiment_id": identifier, **{k: metadata[k] for k in metadata_keys},
                  "condition": "clean", "strength": 0, "input_npz": str(path)}
        manifest_rows.append(record)
        output = RESULTS / f"{identifier}.json"
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            uncertainty = payload.get("groupwise_uncertainty", {})
            result_rows.append({"experiment_id": identifier, "model": metadata["model"],
                                "dataset": metadata["dataset"], "samples": sample_count,
                                "dimension": dimension, **{k: payload["result"][k] for k in
                                ["A_accessibility", "B_separation", "C_information_purity", "NSI",
                                 "geometry_r2", "appearance_r2", "OSR"]},
                                "group_NSI_mean": uncertainty.get("group_balanced_NSI_mean"),
                                "group_NSI_CI_low": uncertainty.get("confidence_interval", [None, None])[0],
                                "group_NSI_CI_high": uncertainty.get("confidence_interval", [None, None])[1],
                                "test_groups": payload["result"]["split_group_counts"]["test"],
                                "result_json": str(output)})
    for path, rows in [(ROOT / "manifests/hpatches_nsi_manifest.csv", manifest_rows),
                       (RESULTS / "hpatches_nsi_summary.csv", result_rows)]:
        if rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["features", "scores", "summary"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--wait-for-features", action="store_true")
    args = parser.parse_args()
    if args.mode == "summary":
        summarize()
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    for identifier, adapter_arguments in MODELS:
        if args.only and identifier not in args.only:
            continue
        path = FEATURES / f"{identifier}.npz"
        if args.mode == "features":
            if path.is_file():
                print(f"Existing complete feature dataset: {identifier}", flush=True)
                continue
            run_command(identifier, "features", ["build-hpatches-model", *adapter_arguments,
                "--export-id", identifier, "--pairs-manifest", str(DATA / "manifests/hpatches/hpatches_pairs.csv"),
                "--dataset-root", str(DATA / "datasets/HPatches/hpatches-sequences-release"),
                "--cache-root", str(DATA / "cache"), "--output", str(path),
                "--max-samples-per-pair", "64", "--seed", "20261023", "--expected-pairs", "580",
                "--proxy", "ncc", "--device", "cuda"])
        else:
            deadline = time.monotonic() + 7200
            while not path.is_file() and args.wait_for_features and time.monotonic() < deadline:
                time.sleep(15)
            if not path.is_file():
                print(f"Feature dataset pending: {identifier}", flush=True)
                continue
            output = RESULTS / f"{identifier}.json"
            if output.is_file():
                existing = json.loads(output.read_text(encoding="utf-8"))
                if "groupwise_uncertainty" in existing:
                    print(f"Existing complete NSI result: {identifier}", flush=True)
                    continue
                # Keep the initial audited point-only result as a recoverable artifact.
                backup = output.with_name(output.stem + "_point_only.json")
                if not backup.exists():
                    backup.write_bytes(output.read_bytes())
            run_command(identifier, "nsi", ["compute-nsi", "--input", str(path),
                "--output", str(output), "--projectors", str(RESULTS / "projectors" / f"{identifier}.npz"),
                "--config", "configs/default.toml"])
        summarize()
    summarize()


if __name__ == "__main__":
    main()
