from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np

from .config import SplitConfig


@dataclass(frozen=True)
class FeatureDataset:
    features: np.ndarray
    geometry: np.ndarray
    appearance: np.ndarray
    groups: np.ndarray
    split: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        arrays = {
            "features": self.features,
            "geometry": self.geometry,
            "appearance": self.appearance,
        }
        n = int(self.features.shape[0])
        for name, value in arrays.items():
            if value.ndim != 2:
                raise ValueError(f"{name} must be two-dimensional; received {value.shape}")
            if value.shape[0] != n:
                raise ValueError(f"{name} has {value.shape[0]} rows; expected {n}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        if n < 12:
            raise ValueError("At least 12 aligned samples are required")
        if self.groups.ndim != 1 or self.groups.shape[0] != n:
            raise ValueError("groups must be a one-dimensional array aligned with features")
        if np.unique(self.groups.astype(str)).size < 3:
            raise ValueError("At least three groups are required for leakage-controlled splitting")
        if self.split is not None:
            if self.split.ndim != 1 or self.split.shape[0] != n:
                raise ValueError("split must be a one-dimensional array aligned with features")
            canonical = {_canonical_split(value) for value in self.split.astype(str)}
            if canonical != {"train", "validation", "test"}:
                raise ValueError(
                    "Explicit split must contain train, validation, and test labels; "
                    f"received {sorted(canonical)}"
                )
            _assert_group_isolation(self.groups.astype(str), self.split.astype(str))
        if self.sample_ids is not None and self.sample_ids.shape[0] != n:
            raise ValueError("sample_ids must be aligned with features")


def _canonical_split(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"val", "valid", "validation", "dev"}:
        return "validation"
    if lowered in {"train", "training"}:
        return "train"
    if lowered in {"test", "testing", "evaluation", "eval"}:
        return "test"
    return lowered


def _assert_group_isolation(groups: np.ndarray, split: np.ndarray) -> None:
    seen: dict[str, set[str]] = {}
    for group, label in zip(groups, split, strict=True):
        seen.setdefault(str(group), set()).add(_canonical_split(str(label)))
    overlapping = {group: labels for group, labels in seen.items() if len(labels) > 1}
    if overlapping:
        preview = dict(list(overlapping.items())[:5])
        raise ValueError(f"Group leakage across splits: {preview}")


def deterministic_group_split(
    groups: np.ndarray,
    config: SplitConfig,
    seed: int,
) -> np.ndarray:
    group_values = np.unique(groups.astype(str))
    if group_values.size < 3:
        raise ValueError("At least three groups are required")
    rng = np.random.default_rng(seed)
    shuffled = group_values.copy()
    rng.shuffle(shuffled)
    n_groups = shuffled.size
    n_train = max(1, int(round(config.train_fraction * n_groups)))
    n_validation = max(1, int(round(config.validation_fraction * n_groups)))
    if n_train + n_validation >= n_groups:
        n_train = max(1, n_groups - 2)
        n_validation = 1
    train_groups = set(shuffled[:n_train])
    validation_groups = set(shuffled[n_train : n_train + n_validation])
    labels = np.empty(groups.shape[0], dtype="U10")
    for index, group in enumerate(groups.astype(str)):
        if group in train_groups:
            labels[index] = "train"
        elif group in validation_groups:
            labels[index] = "validation"
        else:
            labels[index] = "test"
    _assert_group_isolation(groups.astype(str), labels)
    return labels


def canonical_split_array(split: np.ndarray) -> np.ndarray:
    return np.asarray([_canonical_split(str(value)) for value in split], dtype="U10")


def load_feature_npz(path: str | Path) -> FeatureDataset:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"features", "geometry", "appearance", "groups"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Missing required arrays in {source}: {missing}")
        metadata: dict[str, Any] | None = None
        if "metadata_json" in archive.files:
            raw = archive["metadata_json"]
            value = str(raw.item()) if raw.ndim == 0 else str(raw[0])
            metadata = json.loads(value)
        dataset = FeatureDataset(
            features=np.asarray(archive["features"], dtype=np.float64),
            geometry=np.asarray(archive["geometry"], dtype=np.float64),
            appearance=np.asarray(archive["appearance"], dtype=np.float64),
            groups=np.asarray(archive["groups"]).astype(str),
            split=(
                np.asarray(archive["split"]).astype(str)
                if "split" in archive.files
                else None
            ),
            sample_ids=(
                np.asarray(archive["sample_ids"]).astype(str)
                if "sample_ids" in archive.files
                else None
            ),
            metadata=metadata,
        )
    dataset.validate()
    return dataset


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

