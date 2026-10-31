from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import hashlib
import json

import numpy as np
from PIL import Image

from .io import sha256_file


HPATCHES_TARGET_INDICES = tuple(range(2, 7))


def _condition(sequence: str) -> str:
    if sequence.startswith("i_"):
        return "illumination"
    if sequence.startswith("v_"):
        return "viewpoint"
    raise ValueError(
        f"HPatches sequence must start with i_ or v_; received {sequence!r}"
    )


def _stratified_sequence_split(
    sequences: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    assignments: dict[str, str] = {}
    for condition in ("illumination", "viewpoint"):
        members = [name for name in sequences if _condition(name) == condition]
        if len(members) < 3:
            raise ValueError(f"At least three {condition} sequences are required")
        ordered = sorted(
            members,
            key=lambda name: hashlib.sha256(
                f"{seed}:{condition}:{name}".encode("utf-8")
            ).hexdigest(),
        )
        n_train = max(1, int(round(train_fraction * len(ordered))))
        n_validation = max(1, int(round(validation_fraction * len(ordered))))
        if n_train + n_validation >= len(ordered):
            n_train = len(ordered) - 2
            n_validation = 1
        for index, name in enumerate(ordered):
            if index < n_train:
                assignments[name] = "train"
            elif index < n_train + n_validation:
                assignments[name] = "validation"
            else:
                assignments[name] = "test"
    return assignments


def build_hpatches_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20261023,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    expected_sequences: int | None = 116,
) -> dict[str, Any]:
    """Validate HPatches and emit deterministic sequence/pair manifests.

    Splits are assigned at sequence level and independently stratified by the
    official illumination/viewpoint prefixes.  Every referenced file is hashed,
    and homographies are parsed and checked before any manifest is written.
    """

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HPatches dataset root does not exist: {root}")
    sequence_paths = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name[:2] in {"i_", "v_"}
    )
    if expected_sequences is not None and len(sequence_paths) != expected_sequences:
        raise ValueError(
            f"Expected {expected_sequences} HPatches sequences; found {len(sequence_paths)}"
        )
    names = [path.name for path in sequence_paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate HPatches sequence names detected")
    split_by_sequence = _stratified_sequence_split(
        names,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    digest_cache: dict[Path, str] = {}
    image_size_cache: dict[Path, tuple[int, int]] = {}

    def file_digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = sha256_file(path).upper()
        return digest_cache[path]

    def image_size(path: Path) -> tuple[int, int]:
        if path not in image_size_cache:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_size_cache[path] = (int(image.width), int(image.height))
        return image_size_cache[path]

    sequence_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    tree_records: list[str] = []
    for sequence_path in sequence_paths:
        sequence = sequence_path.name
        condition = _condition(sequence)
        split = split_by_sequence[sequence]
        reference = sequence_path / "1.ppm"
        required = [reference]
        for target_index in HPATCHES_TARGET_INDICES:
            required.extend(
                [sequence_path / f"{target_index}.ppm", sequence_path / f"H_1_{target_index}"]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"HPatches sequence {sequence} is incomplete; missing {missing[:3]}"
            )

        reference_width, reference_height = image_size(reference)
        reference_sha = file_digest(reference)
        sequence_rows.append(
            {
                "sequence": sequence,
                "condition": condition,
                "split": split,
                "reference_image": reference.relative_to(root).as_posix(),
                "reference_width": reference_width,
                "reference_height": reference_height,
                "reference_sha256": reference_sha,
            }
        )
        tree_records.append(
            f"{reference.relative_to(root).as_posix()}\t{reference.stat().st_size}\t{reference_sha}"
        )

        for target_index in HPATCHES_TARGET_INDICES:
            target = sequence_path / f"{target_index}.ppm"
            homography = sequence_path / f"H_1_{target_index}"
            target_width, target_height = image_size(target)
            matrix = np.loadtxt(homography, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid homography in {homography}")
            determinant = float(np.linalg.det(matrix))
            if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
                raise ValueError(f"Singular homography in {homography}")
            target_sha = file_digest(target)
            homography_sha = file_digest(homography)
            pair_rows.append(
                {
                    "pair_id": f"{sequence}_1_{target_index}",
                    "sequence": sequence,
                    "condition": condition,
                    "split": split,
                    "reference_image": reference.relative_to(root).as_posix(),
                    "target_image": target.relative_to(root).as_posix(),
                    "homography": homography.relative_to(root).as_posix(),
                    "target_index": target_index,
                    "reference_width": reference_width,
                    "reference_height": reference_height,
                    "target_width": target_width,
                    "target_height": target_height,
                    "reference_sha256": reference_sha,
                    "target_sha256": target_sha,
                    "homography_sha256": homography_sha,
                    "homography_determinant": format(determinant, ".17g"),
                }
            )
            tree_records.extend(
                [
                    f"{target.relative_to(root).as_posix()}\t{target.stat().st_size}\t{target_sha}",
                    f"{homography.relative_to(root).as_posix()}\t{homography.stat().st_size}\t{homography_sha}",
                ]
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequence_manifest = output / "hpatches_sequences.csv"
    pair_manifest = output / "hpatches_pairs.csv"
    with sequence_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    with pair_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    unique_tree_records = sorted(set(tree_records))
    tree_sha256 = hashlib.sha256(
        ("\n".join(unique_tree_records) + "\n").encode("utf-8")
    ).hexdigest().upper()
    split_counts = Counter(split_by_sequence.values())
    condition_counts = Counter(_condition(name) for name in names)
    pair_split_counts = Counter(str(row["split"]) for row in pair_rows)
    report: dict[str, Any] = {
        "dataset": "HPatches sequences release",
        "dataset_root": str(root),
        "sequence_count": len(sequence_rows),
        "pair_count": len(pair_rows),
        "image_count": len(sequence_rows) * 6,
        "homography_count": len(pair_rows),
        "condition_sequence_counts": dict(sorted(condition_counts.items())),
        "split_sequence_counts": dict(sorted(split_counts.items())),
        "split_pair_counts": dict(sorted(pair_split_counts.items())),
        "split_seed": seed,
        "split_strategy": "sequence-isolated SHA-256 ordering stratified by condition",
        "train_fraction_requested": train_fraction,
        "validation_fraction_requested": validation_fraction,
        "test_fraction_requested": round(
            1.0 - train_fraction - validation_fraction, 12
        ),
        "tree_sha256": tree_sha256,
        "sequence_manifest": str(sequence_manifest.resolve()),
        "sequence_manifest_sha256": sha256_file(sequence_manifest).upper(),
        "pair_manifest": str(pair_manifest.resolve()),
        "pair_manifest_sha256": sha256_file(pair_manifest).upper(),
    }
    report_path = output / "hpatches_integrity.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["integrity_report"] = str(report_path.resolve())
    report["integrity_report_sha256"] = sha256_file(report_path).upper()
    return report
