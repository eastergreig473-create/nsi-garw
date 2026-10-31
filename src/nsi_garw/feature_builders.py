from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import os
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm

from .adapters import LocalFeatureAdapter, LocalFeatures, validate_model_metadata
from .attributes import (
    sobel_magnitude,
    structure_tensor,
    tensor_cosine_proxy,
    tensor_ncc_proxy,
    warp_structure_tensor,
)
from .io import FeatureDataset, sha256_file


def _atomic_savez(path: Path, *, compressed: bool = False, **arrays: Any) -> None:
    """Write an NPZ cache atomically so an interrupted run cannot corrupt it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_local_features(path: Path) -> LocalFeatures:
    with np.load(path, allow_pickle=False) as archive:
        metadata_raw = archive["metadata_json"]
        metadata = json.loads(str(metadata_raw.item()))
        features = LocalFeatures(
            coordinates_xy=np.asarray(archive["coordinates_xy"], dtype=np.float64),
            descriptors=np.asarray(archive["descriptors"], dtype=np.float64),
            scores=np.asarray(archive["scores"], dtype=np.float64),
            metadata=metadata,
        )
    features.validate()
    validate_model_metadata(features.metadata)
    return features


def _save_local_features(path: Path, features: LocalFeatures) -> None:
    _atomic_savez(
        path,
        coordinates_xy=np.asarray(features.coordinates_xy, dtype=np.float32),
        descriptors=np.asarray(features.descriptors, dtype=np.float32),
        scores=np.asarray(features.scores, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(features.metadata, ensure_ascii=False, sort_keys=True)
        ),
    )


def _deterministic_subset(
    count: int, maximum: int, *, seed: int, pair_id: str
) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("max_samples_per_pair must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    pair_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(pair_seed)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def build_hpatches_feature_dataset(
    adapter: LocalFeatureAdapter,
    pairs_manifest: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    cache_root: str | Path,
    *,
    export_id: str,
    max_samples_per_pair: int = 64,
    seed: int = 20261023,
    expected_pairs: int | None = 580,
    proxy: str = "ncc",
    allow_empty_pairs: bool = False,
) -> dict[str, Any]:
    """Build one canonical NSI dataset from all HPatches target images.

    The adapter is instantiated once. Features and attribute maps are cached
    separately, and only a deterministic fixed-size subset of valid target
    features enters each image pair so high-resolution dense models cannot
    dominate the sample count.
    """

    root = Path(dataset_root).resolve()
    manifest_path = Path(pairs_manifest).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_id",
        "sequence",
        "split",
        "reference_image",
        "target_image",
        "homography",
    }
    missing_columns = sorted(required.difference(rows[0] if rows else {}))
    if missing_columns:
        raise ValueError(f"HPatches pair manifest is missing columns: {missing_columns}")
    if expected_pairs is not None and len(rows) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} HPatches pairs; found {len(rows)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("HPatches pair_id values must be unique")

    cache = Path(cache_root).resolve()
    feature_cache = cache / "features" / export_id
    attribute_cache = cache / "attributes" / f"hpatches_{proxy}"
    feature_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    appearance_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    canonical_metadata: dict[str, Any] | None = None
    feature_cache_hits = 0
    attribute_cache_hits = 0
    rejected_samples = 0
    reference_tensor_cache: dict[str, np.ndarray] = {}
    empty_pairs: list[str] = []

    for row in tqdm(rows, desc=f"HPatches {export_id}", unit="pair", dynamic_ncols=True, mininterval=10):
        pair_id = row["pair_id"]
        sequence = row["sequence"]
        reference = root / row["reference_image"]
        target = root / row["target_image"]
        homography = root / row["homography"]
        for path in (reference, target, homography):
            if not path.is_file():
                raise FileNotFoundError(path)

        target_stem = Path(row["target_image"]).stem
        local_path = feature_cache / sequence / f"{target_stem}.npz"
        target_image_array: np.ndarray | None = None
        local: LocalFeatures | None = None
        if local_path.is_file() and not zipfile.is_zipfile(local_path):
            local_path.unlink(missing_ok=True)
        if local_path.is_file():
            try:
                local = _load_local_features(local_path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                local_path.unlink(missing_ok=True)
            else:
                feature_cache_hits += 1
        if local is None:
            target_image_array = np.asarray(Image.open(target).convert("RGB"))
            local = adapter.extract(target_image_array)
            _save_local_features(local_path, local)
        if canonical_metadata is None:
            canonical_metadata = dict(local.metadata)
        elif local.metadata != canonical_metadata:
            raise ValueError(f"Adapter metadata changed within export {export_id}")

        attribute_path = attribute_cache / sequence / f"{pair_id}.npz"
        # A byte-verified SSD copy reduces repeated reads from the USB data disk.
        # The canonical cache and all scientific inputs remain on F:.
        hot = Path(__file__).resolve().parents[2] / ".cache/attribute_hot" / f"hpatches_{proxy}" / sequence / attribute_path.name
        if hot.is_file() and attribute_path.is_file():
            original_stat, hot_stat = attribute_path.stat(), hot.stat()
            if original_stat.st_size == hot_stat.st_size and original_stat.st_mtime_ns == hot_stat.st_mtime_ns:
                attribute_path = hot
        cached_attributes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if attribute_path.is_file() and not zipfile.is_zipfile(attribute_path):
            attribute_path.unlink(missing_ok=True)
        if attribute_path.is_file():
            try:
                with np.load(attribute_path, allow_pickle=False) as attributes:
                    cached_attributes = (
                        np.asarray(attributes["geometry"], dtype=np.float64)[..., 0],
                        np.asarray(attributes["appearance"], dtype=np.float64)[..., 0],
                        np.asarray(attributes["geometry_valid"], dtype=bool),
                    )
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                attribute_path.unlink(missing_ok=True)
            else:
                attribute_cache_hits += 1
        if cached_attributes is None:
            if sequence not in reference_tensor_cache:
                reference_image = np.asarray(Image.open(reference).convert("RGB"))
                reference_tensor_cache[sequence] = structure_tensor(reference_image)
            if target_image_array is None:
                target_image_array = np.asarray(Image.open(target).convert("RGB"))
            target_tensor = structure_tensor(target_image_array)
            matrix = np.loadtxt(homography, dtype=np.float64)
            transferred, valid = warp_structure_tensor(
                reference_tensor_cache[sequence], matrix, target_image_array.shape[:2]
            )
            if proxy == "ncc":
                geometry_map = tensor_ncc_proxy(transferred, target_tensor, valid)
            elif proxy == "cosine":
                geometry_map = tensor_cosine_proxy(transferred, target_tensor, valid)
            else:
                raise ValueError("proxy must be 'ncc' or 'cosine'")
            appearance_map = sobel_magnitude(target_image_array)
            valid_map = np.isfinite(geometry_map) & valid
            attribute_metadata = {
                "annotation_type": "hpatches_homography_structure_tensor_compact",
                "geometry_proxy": proxy,
                "reference_image": row["reference_image"],
                "target_image": row["target_image"],
                "homography": row["homography"],
                "reference_sha256": row.get("reference_sha256", ""),
                "target_sha256": row.get("target_sha256", ""),
                "homography_sha256": row.get("homography_sha256", ""),
            }
            _atomic_savez(
                attribute_path,
                geometry=geometry_map[..., None].astype(np.float32),
                geometry_valid=valid_map,
                appearance=appearance_map[..., None].astype(np.float32),
                metadata_json=np.asarray(
                    json.dumps(attribute_metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
        else:
            geometry_map, appearance_map, valid_map = cached_attributes

        xy = np.asarray(local.coordinates_xy, dtype=np.float64)
        inside = (
            (xy[:, 0] >= 0.0)
            & (xy[:, 0] <= geometry_map.shape[1] - 1.0)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] <= geometry_map.shape[0] - 1.0)
        )
        sample_coordinates = [xy[:, 1], xy[:, 0]]
        geometry = map_coordinates(
            geometry_map, sample_coordinates, order=1, mode="constant", cval=np.nan
        )
        appearance = map_coordinates(
            appearance_map, sample_coordinates, order=1, mode="constant", cval=np.nan
        )
        valid = map_coordinates(
            valid_map.astype(np.float32),
            sample_coordinates,
            order=0,
            mode="constant",
            cval=0.0,
        ) > 0.5
        keep = inside & valid & np.isfinite(geometry) & np.isfinite(appearance)
        valid_indices = np.flatnonzero(keep)
        rejected_samples += int(len(keep) - len(valid_indices))
        if len(valid_indices) == 0:
            if allow_empty_pairs:
                empty_pairs.append(pair_id)
                continue
            raise ValueError(f"No valid aligned features for HPatches pair {pair_id}")
        selected_local = _deterministic_subset(
            len(valid_indices),
            max_samples_per_pair,
            seed=seed,
            pair_id=pair_id,
        )
        selected = valid_indices[selected_local]
        feature_parts.append(
            np.asarray(local.descriptors[selected], dtype=np.float32)
        )
        geometry_parts.append(geometry[selected, None].astype(np.float32))
        appearance_parts.append(appearance[selected, None].astype(np.float32))
        group_parts.append(np.full(len(selected), sequence, dtype="U128"))
        split_parts.append(np.full(len(selected), row["split"], dtype="U10"))
        sample_id_parts.append(
            np.asarray([f"{pair_id}:{index}" for index in selected], dtype="U192")
        )

    if canonical_metadata is None:
        raise ValueError("HPatches pair manifest contains no rows")
    dataset_metadata = {
        **canonical_metadata,
        "dataset": "HPatches sequences release",
        "annotation_source": f"homography structure-tensor {proxy} plus Sobel magnitude",
        "split_id": f"hpatches-sequence-stratified-seed-{seed}",
        "pairs_manifest": str(manifest_path),
        "pairs_manifest_sha256": sha256_file(manifest_path).upper(),
        "sampling": (
            f"up to {max_samples_per_pair} valid features per pair selected by "
            f"SHA-256-seeded sampling; seed {seed}"
        ),
        "export_id": export_id,
        "empty_pair_policy": "omit from conditional NSI; retain in matching denominator" if allow_empty_pairs else "error",
        "empty_pairs": empty_pairs,
    }
    dataset = FeatureDataset(
        features=np.concatenate(feature_parts, axis=0),
        geometry=np.concatenate(geometry_parts, axis=0),
        appearance=np.concatenate(appearance_parts, axis=0),
        groups=np.concatenate(group_parts, axis=0),
        split=np.concatenate(split_parts, axis=0),
        sample_ids=np.concatenate(sample_id_parts, axis=0),
        metadata=dataset_metadata,
    )
    dataset.validate()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        destination,
        compressed=True,
        features=np.asarray(dataset.features, dtype=np.float32),
        geometry=np.asarray(dataset.geometry, dtype=np.float32),
        appearance=np.asarray(dataset.appearance, dtype=np.float32),
        groups=dataset.groups,
        split=dataset.split,
        sample_ids=dataset.sample_ids,
        metadata_json=np.asarray(
            json.dumps(dataset.metadata, ensure_ascii=False, sort_keys=True)
        ),
    )
    return {
        "output": str(destination.resolve()),
        "output_sha256": sha256_file(destination).upper(),
        "export_id": export_id,
        "pair_count": len(rows),
        "pairs_with_aligned_features": len(rows) - len(empty_pairs),
        "empty_pairs": empty_pairs,
        "sample_count": int(dataset.features.shape[0]),
        "feature_dimension": int(dataset.features.shape[1]),
        "group_count": int(np.unique(dataset.groups).size),
        "split_counts": {
            label: int(np.sum(dataset.split == label))
            for label in ("train", "validation", "test")
        },
        "feature_cache_hits": feature_cache_hits,
        "attribute_cache_hits": attribute_cache_hits,
        "rejected_unaligned_samples": rejected_samples,
        "metadata": dataset_metadata,
    }
