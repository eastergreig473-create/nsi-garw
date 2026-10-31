from pathlib import Path

import numpy as np
from PIL import Image

from nsi_garw.datasets import build_hpatches_manifests


def _write_sequence(root: Path, name: str) -> None:
    sequence = root / name
    sequence.mkdir()
    for index in range(1, 7):
        image = np.full((8 + index, 10 + index, 3), index * 20, dtype=np.uint8)
        Image.fromarray(image).save(sequence / f"{index}.ppm")
    for index in range(2, 7):
        np.savetxt(sequence / f"H_1_{index}", np.eye(3), fmt="%.8f")


def test_build_hpatches_manifests_is_deterministic_and_leakage_free(tmp_path: Path) -> None:
    root = tmp_path / "hpatches"
    root.mkdir()
    for prefix in ("i", "v"):
        for index in range(4):
            _write_sequence(root, f"{prefix}_scene_{index}")

    first = build_hpatches_manifests(
        root, tmp_path / "out-a", seed=17, expected_sequences=8
    )
    second = build_hpatches_manifests(
        root, tmp_path / "out-b", seed=17, expected_sequences=8
    )

    assert first["sequence_count"] == 8
    assert first["pair_count"] == 40
    assert first["image_count"] == 48
    assert first["tree_sha256"] == second["tree_sha256"]
    assert first["sequence_manifest_sha256"] == second["sequence_manifest_sha256"]
    assert first["pair_manifest_sha256"] == second["pair_manifest_sha256"]

    rows = (tmp_path / "out-a" / "hpatches_pairs.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 41
    by_sequence: dict[str, set[str]] = {}
    header = rows[0].split(",")
    sequence_index = header.index("sequence")
    split_index = header.index("split")
    for row in rows[1:]:
        fields = row.split(",")
        by_sequence.setdefault(fields[sequence_index], set()).add(fields[split_index])
    assert all(len(labels) == 1 for labels in by_sequence.values())
