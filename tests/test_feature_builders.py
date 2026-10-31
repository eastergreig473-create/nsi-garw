from pathlib import Path
import csv
from dataclasses import replace

import numpy as np
from PIL import Image

from nsi_garw.adapters import LocalFeatures
from nsi_garw.feature_builders import build_hpatches_feature_dataset, _save_local_features
from nsi_garw.io import load_feature_npz


class _GridAdapter:
    name = "grid"

    def extract(self, image: np.ndarray) -> LocalFeatures:
        height, width = image.shape[:2]
        y, x = np.meshgrid(np.arange(4, height - 4, 4), np.arange(4, width - 4, 4), indexing="ij")
        coordinates = np.stack([x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float64)
        descriptors = np.stack(
            [coordinates[:, 0] / width, coordinates[:, 1] / height], axis=1
        )
        return LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=np.ones(len(coordinates), dtype=np.float64),
            metadata={
                "model": "grid",
                "implementation": "test",
                "implementation_commit": "test-commit",
                "checkpoint": "none",
                "checkpoint_sha256": "none",
                "layer": "grid",
                "preprocessing": "none",
                "coordinate_system": "original-image xy",
            },
        )


def test_build_hpatches_feature_dataset_aligns_and_caches(tmp_path: Path) -> None:
    root = tmp_path / "hpatches"
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for sequence_index in range(2):
            sequence = f"i_{split}_{sequence_index}"
            sequence_path = root / sequence
            sequence_path.mkdir(parents=True)
            reference = np.zeros((32, 40, 3), dtype=np.uint8)
            reference[:, 10:30] = 180
            target = np.roll(reference, split_index + 1, axis=1)
            Image.fromarray(reference).save(sequence_path / "1.ppm")
            Image.fromarray(target).save(sequence_path / "2.ppm")
            homography = np.eye(3)
            np.savetxt(sequence_path / "H_1_2", homography)
            rows.append(
                {
                    "pair_id": f"{sequence}_1_2",
                    "sequence": sequence,
                    "split": split,
                    "reference_image": f"{sequence}/1.ppm",
                    "target_image": f"{sequence}/2.ppm",
                    "homography": f"{sequence}/H_1_2",
                }
            )
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first = build_hpatches_feature_dataset(
        _GridAdapter(),
        manifest,
        root,
        tmp_path / "dataset-a.npz",
        tmp_path / "cache",
        export_id="grid",
        max_samples_per_pair=4,
        expected_pairs=6,
    )
    second = build_hpatches_feature_dataset(
        _GridAdapter(),
        manifest,
        root,
        tmp_path / "dataset-b.npz",
        tmp_path / "cache",
        export_id="grid",
        max_samples_per_pair=4,
        expected_pairs=6,
    )
    dataset = load_feature_npz(tmp_path / "dataset-a.npz")
    cached_dataset = load_feature_npz(tmp_path / "dataset-b.npz")
    assert first["sample_count"] == 24
    assert dataset.features.shape == (24, 2)
    assert set(dataset.split.tolist()) == {"train", "validation", "test"}
    assert second["feature_cache_hits"] == 6
    assert second["attribute_cache_hits"] == 6
    np.testing.assert_allclose(dataset.features, cached_dataset.features)
    np.testing.assert_allclose(dataset.geometry, cached_dataset.geometry)
    np.testing.assert_allclose(dataset.appearance, cached_dataset.appearance)
    np.testing.assert_array_equal(dataset.groups, cached_dataset.groups)
    np.testing.assert_array_equal(dataset.split, cached_dataset.split)
    np.testing.assert_array_equal(dataset.sample_ids, cached_dataset.sample_ids)
    assert dataset.metadata == cached_dataset.metadata

    empty = _GridAdapter().extract(np.zeros((32, 40, 3), dtype=np.uint8))
    empty = replace(empty, coordinates_xy=empty.coordinates_xy[:0],
                    descriptors=empty.descriptors[:0], scores=empty.scores[:0])
    _save_local_features(tmp_path / "cache/features/grid" / rows[0]["sequence"] / "2.npz", empty)
    conditional = build_hpatches_feature_dataset(
        _GridAdapter(), manifest, root, tmp_path / "conditional.npz", tmp_path / "cache",
        export_id="grid", max_samples_per_pair=4, expected_pairs=6, allow_empty_pairs=True,
    )
    assert conditional["empty_pairs"] == [rows[0]["pair_id"]]
    assert conditional["pairs_with_aligned_features"] == 5
    assert conditional["sample_count"] == 20
