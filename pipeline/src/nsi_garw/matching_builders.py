from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .adapters import LocalFeatureAdapter, LocalFeatures
from .evaluation import homography_reprojection_errors
from .feature_builders import _load_local_features, _save_local_features
from .io import sha256_file


def mutual_nearest_matches(first: LocalFeatures, second: LocalFeatures) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbors in row-normalized descriptor Euclidean distance."""
    import cv2

    if first.descriptors.shape[1] != second.descriptors.shape[1]:
        raise ValueError("Descriptor dimensions must agree")
    if not len(first.descriptors) or not len(second.descriptors):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    x = np.asarray(first.descriptors, dtype=np.float32)
    y = np.asarray(second.descriptors, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(x, y)
    matches = sorted(matches, key=lambda item: (item.queryIdx, item.trainIdx))
    return (np.asarray([m.queryIdx for m in matches], dtype=np.int64),
            np.asarray([m.trainIdx for m in matches], dtype=np.int64))


def pair_matching_record(
    first: LocalFeatures, second: LocalFeatures, homography: np.ndarray,
    *, pair_id: str, sequence: str, split: str,
    thresholds: tuple[float, ...] = (1., 3., 5., 10.),
) -> dict[str, Any]:
    indices_first, indices_second = mutual_nearest_matches(first, second)
    errors = homography_reprojection_errors(
        first.coordinates_xy[indices_first], second.coordinates_xy[indices_second],
        homography, symmetric=True,
    )
    # Empty-match pairs stay in the denominator with zero MMA.  Singular
    # projections count as failures rather than disappearing from the mean.
    count = len(errors)
    finite = np.isfinite(errors)
    record = {"pair_id": pair_id, "scene": sequence, "split": split,
              "source_feature_count": len(first.descriptors),
              "target_feature_count": len(second.descriptors), "match_count": count,
              "invalid_projection_count": int(np.sum(~finite)),
              "median_reprojection_error": float(np.median(errors[finite])) if finite.any() else None}
    for threshold in thresholds:
        record[f"mma_at_{threshold:g}"] = float(np.sum(finite & (errors <= threshold)) / count) if count else 0.
    return record


def summarize_pair_matching(
    records: list[dict[str, Any]], *, seed: int = 20261023,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    table = pd.DataFrame(records)
    columns = [c for c in table.columns if c.startswith("mma_at_")]
    results = {}
    for label, subset in [("all_sequences_descriptive", table),
                          ("held_out_test_sequences", table[table["split"] == "test"])]:
        if subset.empty:
            continue
        per_scene = subset.groupby("scene")[columns].mean()
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(per_scene), (bootstrap_replicates, len(per_scene)))
        metrics = {}
        for column in columns:
            values = per_scene[column].to_numpy(dtype=float)
            ci = np.percentile(values[draw].mean(axis=1), [2.5, 97.5])
            metrics[column] = {"estimate": float(values.mean()), "confidence_interval": ci.tolist()}
        results[label] = {"sequence_count": len(per_scene), "pair_count": len(subset),
                          "zero_match_pairs": int(np.sum(subset["match_count"] == 0)),
                          "bootstrap_replicates": bootstrap_replicates, "metrics": metrics}
    return results


def build_hpatches_descriptor_matching(
    adapter: LocalFeatureAdapter, pairs_manifest: str | Path, dataset_root: str | Path,
    cache_root: str | Path, output_directory: str | Path, *, export_id: str,
) -> dict[str, Any]:
    root, cache = Path(dataset_root), Path(cache_root) / "features" / export_id
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with Path(pairs_manifest).open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    records = []
    metadata = None
    for row in tqdm(pairs, desc=f"MNN {export_id}", unit="pair", mininterval=5):
        features = []
        for field in ["reference_image", "target_image"]:
            relative = Path(row[field])
            path = cache / row["sequence"] / f"{relative.stem}.npz"
            if path.exists():
                local = _load_local_features(path)
            else:
                local = adapter.extract(np.asarray(Image.open(root / relative).convert("RGB")))
                _save_local_features(path, local)
            features.append(local)
        if features[0].metadata != features[1].metadata:
            raise ValueError(f"Source/target extraction metadata differ for {row['pair_id']}")
        metadata = features[0].metadata
        record = pair_matching_record(*features, np.loadtxt(root / row["homography"]),
            pair_id=row["pair_id"], sequence=row["sequence"], split=row["split"])
        records.append(record)
    pd.DataFrame(records).to_csv(output / "per_pair.csv", index=False)
    payload = {"export_id": export_id, "method": "row-normalized descriptors + mutual nearest neighbors",
               "scope": "descriptor benchmark; does not evaluate a paired model decoder or matcher",
               "error": "RMS symmetric homography reprojection error in original-image pixels",
               "zero_match_policy": "MMA=0; retain pair and sequence",
               "pairs_manifest_sha256": sha256_file(pairs_manifest),
               "model_metadata": metadata, "results": summarize_pair_matching(records)}
    (output / "matching_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
