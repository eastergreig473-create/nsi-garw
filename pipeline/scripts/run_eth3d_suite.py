"""Scene-isolated ETH3D NSI exports from CRC-verified calibrated laser depth."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, sobel

from nsi_garw.attributes import rgb_to_gray
from nsi_garw.cli import _batch_adapter, build_parser
from nsi_garw.config import load_config
from nsi_garw.eth3d import read_cameras, read_images
from nsi_garw.experiments import compute_nsi_artifacts, source_sha256
from nsi_garw.feature_builders import _atomic_savez, _load_local_features, _save_local_features, _deterministic_subset
from nsi_garw.io import FeatureDataset, load_feature_npz, sha256_file
from run_hpatches_suite import DATA, MODELS, ROOT

PROTOCOL = "eth3d_pinhole_laser_normals3_signed_sobel3_v1"


def sampled_normals(depth, coordinates, intrinsics):
    """Central-difference normals at nearest pixels; COLMAP centers are i+0.5."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    if fx <= 0 or fy <= 0:
        raise ValueError("Invalid focal length")
    count = len(coordinates)
    output = np.zeros((count, 3), dtype=np.float64)
    finite = np.isfinite(coordinates).all(axis=1)
    indices = np.zeros((count, 2), dtype=np.int64)
    indices[finite] = np.floor(coordinates[finite] + .5).astype(np.int64)
    x, y = indices.T
    inside = finite & (x >= 1) & (x < depth.shape[1]-1) & (y >= 1) & (y < depth.shape[0]-1)
    selected = np.flatnonzero(inside)
    x, y = x[selected], y[selected]
    z = np.stack([depth[y, x], depth[y, x-1], depth[y, x+1], depth[y-1, x], depth[y+1, x]], axis=1)
    valid_depth = np.isfinite(z).all(axis=1) & (z > 0).all(axis=1)
    selected, x, y, z = selected[valid_depth], x[valid_depth], y[valid_depth], z[valid_depth]
    def points(xx, yy, zz):
        return np.column_stack([(xx+.5-cx)*zz/fx, (yy+.5-cy)*zz/fy, zz])
    dx = points(x+1, y, z[:, 2]) - points(x-1, y, z[:, 1])
    dy = points(x, y+1, z[:, 4]) - points(x, y-1, z[:, 3])
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=1)
    stable = np.isfinite(normals).all(axis=1) & (lengths > 1e-12)
    normals = normals[stable] / lengths[stable, None]
    normals[normals[:, 2] < 0] *= -1
    selected = selected[stable]
    output[selected] = normals
    keep = np.zeros(count, dtype=bool)
    keep[selected] = True
    return output, keep


def appearance_cache(image, image_sha, path):
    """Cache full-resolution float32 signed Sobel channels and magnitude on SSD."""
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record["protocol"] == PROTOCOL and record["image_sha256"] == image_sha:
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            if cached.shape == (*image.shape[:2], 3) and cached.dtype == np.float32:
                return cached
    gray = rgb_to_gray(image)
    horizontal = sobel(gray, axis=1, mode="reflect")
    vertical = sobel(gray, axis=0, mode="reflect")
    magnitude = np.hypot(horizontal, vertical)
    scale = max(float(np.percentile(magnitude, 99.)), 1e-12)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.npy")
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(*image.shape[:2], 3))
    for channel, values, low in ((0, horizontal, -1.), (1, vertical, -1.), (2, magnitude, 0.)):
        cache[..., channel] = np.clip(values / scale, low, 1.)
    cache.flush()
    del cache
    temporary.replace(path)
    sidecar.write_text(json.dumps({"protocol": PROTOCOL, "image_sha256": image_sha,
        "magnitude_percentile99": scale, "dtype": "float32", "cache_sha256": sha256_file(path)}, indent=2), encoding="utf-8")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def export(identifier, adapter_args, records, manifest, split_path, args):
    export_id = "eth3d_" + identifier
    destination = DATA / "manifests/features" / (export_id + ".npz")
    runner_sha = sha256_file(Path(__file__))
    manifest_sha, split_sha = sha256_file(manifest), sha256_file(split_path)
    report_path = ROOT / "results/real/eth3d" / (export_id + ".build.json")
    if destination.exists():
        dataset = load_feature_npz(destination)
        expected = {"runner_sha256": runner_sha, "image_manifest_sha256": manifest_sha,
                    "split_manifest_sha256": split_sha, "protocol": PROTOCOL}
        if any(dataset.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing ETH3D export has incompatible provenance: {destination}")
        del dataset
        return destination
    arguments = build_parser().parse_args(["build-hpatches-model", *adapter_args,
        "--export-id", export_id, "--pairs-manifest", str(manifest),
        "--dataset-root", str(args.root), "--cache-root", str(DATA / "cache"),
        "--output", str(destination), "--device", "cuda"])
    adapter = _batch_adapter(arguments)
    camera_cache, image_cache, metadata = {}, {}, None
    parts = {name: [] for name in ("features", "geometry", "appearance", "groups", "split", "sample_ids")}
    coverage = []
    for index, record in enumerate(records):
        if record["state"] != "prepared":
            coverage.append({"scene": record["scene"], "image": record["image"], "state": record["state"], "samples": 0})
            continue
        scene = record["scene"]
        image_path, depth_path = Path(record["image"]), Path(record["output"])
        if sha256_file(image_path) != record["image_sha256"] or sha256_file(depth_path) != record["output_sha256"]:
            raise ValueError(f"Calibrated ETH3D input changed: {image_path}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        if depth.shape != image.shape[:2]:
            raise ValueError("Depth and image shape differ")
        calibration = args.root / scene / "dslr_calibration_undistorted"
        if scene not in camera_cache:
            camera_cache[scene] = read_cameras(calibration / "cameras.txt")
            image_cache[scene] = read_images(calibration / "images.txt")
        camera_id = image_cache[scene][image_path.name]["camera_id"]
        camera = camera_cache[scene][camera_id]
        if camera.model_name != "PINHOLE":
            raise ValueError("ETH3D target calibration is not PINHOLE")
        local_path = DATA / "cache/features" / export_id / scene / (image_path.name + ".npz")
        if local_path.exists():
            local = _load_local_features(local_path)
        else:
            local = adapter.extract(image)
            _save_local_features(local_path, local)
        if metadata is None:
            metadata = dict(local.metadata)
        elif local.metadata != metadata:
            raise ValueError("Adapter preprocessing changed within the ETH3D export")
        normals, valid = sampled_normals(depth, local.coordinates_xy, camera.params)
        appearance = appearance_cache(image, record["image_sha256"],
            ROOT / ".cache/eth3d_attributes" / scene / (image_path.name + ".npy"))
        coordinates = [local.coordinates_xy[:, 1], local.coordinates_xy[:, 0]]
        sampled = np.column_stack([map_coordinates(appearance[..., channel], coordinates,
            order=1, mode="constant", cval=np.nan) for channel in range(3)])
        valid &= np.isfinite(sampled).all(axis=1)
        indices = np.flatnonzero(valid)
        chosen = indices[_deterministic_subset(len(indices), args.samples_per_image,
            seed=20261023, pair_id=scene + ":" + image_path.name)]
        coverage.append({"scene": scene, "image": str(image_path), "state": "sampled" if len(chosen) else "no_valid_aligned_features",
                         "native_feature_count": len(local.descriptors), "valid_feature_count": len(indices), "samples": len(chosen)})
        if len(chosen):
            parts["features"].append(local.descriptors[chosen].astype(np.float32))
            parts["geometry"].append(normals[chosen].astype(np.float32))
            parts["appearance"].append(sampled[chosen].astype(np.float32))
            parts["groups"].append(np.full(len(chosen), scene, dtype="U32"))
            parts["split"].append(np.full(len(chosen), record["split"], dtype="U10"))
            parts["sample_ids"].append(np.asarray([f"{scene}:{image_path.name}:{i}" for i in chosen], dtype="U128"))
        if (index+1) % 10 == 0:
            print(f"ETH3D {identifier}: {index+1}/{len(records)} images", flush=True)
        del image, depth, local, normals, appearance, sampled
    metadata.update({"dataset": "ETH3D high-resolution public training scenes", "protocol": PROTOCOL,
        "annotation_source": "calibrated laser depth camera-frame normals (3 outputs); signed Sobel x/y and magnitude (3 outputs)",
        "split_id": "eth3d-public13-indoor-outdoor-stratified-5-3-5-seed-20261023",
        "sampling": f"up to {args.samples_per_image} valid native features per image; SHA256-seeded; no empty-image replacement",
        "normal_stencil": "nearest original-resolution pixel; central difference in 3D; all five depth values finite and positive; COLMAP corner-origin centers i+0.5; camera-z sign canonical",
        "appearance_normalization": "all three channels divided by image-wise magnitude percentile99; x/y clipped to [-1,1], magnitude to [0,1]; native-resolution float32 cache",
        "runner_sha256": runner_sha, "source_sha256": source_sha256(), "image_manifest_sha256": manifest_sha,
        "split_manifest_sha256": split_sha, "export_id": export_id})
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    FeatureDataset(**arrays, metadata=metadata).validate()
    if set(np.unique(arrays["groups"])) != set(r["scene"] for r in records):
        raise ValueError("At least one ETH3D scene has no valid sampled features")
    _atomic_savez(destination, compressed=True, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"metadata": metadata, "image_coverage": coverage,
        "sample_count": len(arrays["features"]), "feature_dimension": arrays["features"].shape[1],
        "output_sha256": sha256_file(destination)}, indent=2), encoding="utf-8")
    del arrays, parts, adapter
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=[name for name, _ in MODELS], choices=[name for name, _ in MODELS])
    parser.add_argument("--samples-per-image", type=int, default=128)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    manifest = DATA / "manifests/eth3d/depth_preprocessing/all_images.json"
    split_path = DATA / "manifests/eth3d/frozen_scene_split.json"
    deadline = time.monotonic() + 8*3600
    while not manifest.exists():
        if not args.wait or time.monotonic() > deadline:
            raise FileNotFoundError(manifest)
        time.sleep(20)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    scene_split = {scene: label for label in ("train", "validation", "test") for scene in split[label]}
    if len(scene_split) != 13 or any(scene_split.get(record["scene"]) != record["split"] for record in records):
        raise ValueError("ETH3D input records disagree with the frozen scene split")
    for identifier, adapter_args in MODELS:
        if identifier not in args.models:
            continue
        print(f"ETH3D START {identifier}", flush=True)
        source = export(identifier, adapter_args, records, manifest, split_path, args)
        result = ROOT / "results/real/eth3d" / ("eth3d_" + identifier + ".json")
        if not result.exists():
            compute_nsi_artifacts(source, result, load_config(ROOT / "configs/default.toml"))
        print(f"ETH3D DONE {identifier}", flush=True)


if __name__ == "__main__":
    main()
