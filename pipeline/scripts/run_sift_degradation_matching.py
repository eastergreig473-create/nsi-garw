"""Full 116-sequence SIFT stress test with resumable per-condition artifacts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from nsi_garw.adapters import SIFTAdapter
from nsi_garw.degradations import DEGRADATIONS, apply_degradation
from nsi_garw.experiments import source_sha256
from nsi_garw.feature_builders import _load_local_features, _save_local_features
from nsi_garw.io import sha256_file
from nsi_garw.matching_builders import pair_matching_record, summarize_pair_matching

DATA = Path("F:/NSI_GARW_Data")
ROOT = Path(__file__).resolve().parents[1]
PAIRS = DATA / "manifests/hpatches/hpatches_pairs.csv"
IMAGES = DATA / "datasets/HPatches/hpatches-sequences-release"


def _initialize_worker() -> None:
    import cv2
    cv2.setNumThreads(1)
    global _adapter
    _adapter = SIFTAdapter(max_keypoints=4096)


def _pair(task: tuple[dict[str, str], str, float, str]) -> dict:
    row, kind, strength, identifier = task
    reference_path = DATA / "cache/features/sift_default" / row["sequence"] / "1.npz"
    if reference_path.exists():
        reference = _load_local_features(reference_path)
    else:
        reference = _adapter.extract(np.asarray(Image.open(IMAGES / row["reference_image"]).convert("RGB")))
        _save_local_features(reference_path, reference)
    target_path = DATA / "cache/features" / identifier / row["sequence"] / (Path(row["target_image"]).stem + ".npz")
    if target_path.exists():
        target = _load_local_features(target_path)
    else:
        image = np.asarray(Image.open(IMAGES / row["target_image"]).convert("RGB"))
        seed = int.from_bytes(hashlib.sha256(f"20261023:{row['pair_id']}:{kind}".encode()).digest()[:8], "little")
        altered = apply_degradation(image, kind, strength, seed=seed)
        encoded = np.clip(np.rint(altered * 255.), 0, 255).astype(np.uint8)
        target = _adapter.extract(encoded)
        target.metadata["preprocessing"] += f"; target-only {kind} degradation strength {strength:g}; SHA-256 pair-seeded"
        target.metadata.update({"degradation_kind": kind, "degradation_strength": strength,
                                "degradation_seed_rule": "SHA256 20261023:pair_id:kind first 8 bytes little endian"})
        _save_local_features(target_path, target)
    record = pair_matching_record(reference, target, np.loadtxt(IMAGES / row["homography"]),
        pair_id=row["pair_id"], sequence=row["sequence"], split=row["split"])
    return {"condition": kind, "strength": strength, **record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kinds", nargs="*", default=list(DEGRADATIONS))
    parser.add_argument("--strengths", nargs="*", type=float, default=[.2, .4, .6, .8, 1.])
    args = parser.parse_args()
    os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    with PAIRS.open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    output = ROOT / "results/real/sift_degradation_matching"
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker) as pool:
        for kind in args.kinds:
            if kind not in DEGRADATIONS:
                raise ValueError(kind)
            for strength in args.strengths:
                identifier = f"sift_degraded_{kind}_s{round(strength * 100):03d}"
                folder = output / identifier
                if (folder / "matching_summary.json").exists():
                    print(f"Existing completed condition: {identifier}", flush=True)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                started, source_hash = time.perf_counter(), source_sha256()
                print(f"START {identifier}: {len(pairs)} pairs", flush=True)
                records = []
                tasks = [(row, kind, strength, identifier) for row in pairs]
                for record in pool.map(_pair, tasks, chunksize=2):
                    records.append(record)
                    if len(records) % 100 == 0:
                        print(f"{identifier}: {len(records)}/{len(pairs)}", flush=True)
                pd.DataFrame(records).to_csv(folder / "per_pair.csv", index=False)
                payload = {"condition": kind, "strength": strength, "export_id": identifier,
                    "dataset": "HPatches sequences release", "processed_pair_count": len(records),
                    "method": "SIFT max4096 + row-normalized descriptors + mutual nearest neighbors",
                    "perturbation": "target image only; quantize to RGB uint8 after degradation; reference and homography unchanged",
                    "error": "RMS symmetric homography reprojection error in original-image pixels",
                    "zero_match_policy": "MMA=0; retain pair and sequence",
                    "pairs_manifest_sha256": sha256_file(PAIRS), "source_sha256": source_hash,
                    "seed": 20261023, "wall_seconds": time.perf_counter() - started,
                    "results": summarize_pair_matching(records)}
                (folder / "matching_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"DONE {identifier}: {payload['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
