"""Paired laser/SfM normals on identical tracked-point neighborhoods.

This is a sparse common-support sensitivity analysis, not a dense COLMAP
reconstruction or an independent test of the supplied camera calibration.
All new scientific settings are frozen before computing paired NSI outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts, source_sha256, write_json
from nsi_garw.feature_builders import _atomic_savez
from nsi_garw.io import FeatureDataset, load_feature_npz, sha256_file
from run_hpatches_suite import DATA, MODELS, RESULTS, ROOT


RADIUS = 64.0
MAX_NEIGHBORS = 16
MIN_NEIGHBORS = 6
MAX_EIGEN_RATIO = 0.25
PROTOCOL_ID = "eth3d_laser_sfm_common_tracked_support_r64_k16_min6_plane_v1"


def plane_normal(points):
    """Unweighted TLS plane; reject line-like or poorly planar support."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < MIN_NEIGHBORS or not np.isfinite(points).all():
        return None
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    if eigenvalues[1] <= 1e-12 or eigenvalues[0] / eigenvalues[1] > MAX_EIGEN_RATIO:
        return None
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    return normal


def common_clouds(xyz_camera, camera, depth):
    """Project tracked SfM points, then sample laser z at the same pixel rays.

    COLMAP corner-origin xy is converted to zero-based pixel-center xy.
    Nearest pixels preserve missing laser data. No depth/normal agreement
    filter is applied. Laser points retain the *projected* subpixel ray.
    """
    xyz = np.asarray(xyz_camera, dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz = xyz[valid]
    xy_corner = np.asarray(camera.img_from_cam(xyz))
    inside = np.isfinite(xy_corner).all(axis=1)
    inside &= (xy_corner[:, 0] >= 0) & (xy_corner[:, 0] < depth.shape[1])
    inside &= (xy_corner[:, 1] >= 0) & (xy_corner[:, 1] < depth.shape[0])
    xyz, xy_corner = xyz[inside], xy_corner[inside]
    ij = np.floor(xy_corner).astype(np.int64)
    laser_z = depth[ij[:, 1], ij[:, 0]]
    valid = np.isfinite(laser_z) & (laser_z > 0)
    xyz, xy_corner, laser_z = xyz[valid], xy_corner[valid], laser_z[valid]
    rays = np.asarray(camera.cam_from_img(xy_corner))
    laser_xyz = np.column_stack([rays * laser_z[:, None], laser_z])
    return xy_corner - 0.5, xyz, laser_xyz


def paired_normals(coordinates, support_xy, sfm_cloud, laser_cloud):
    count = len(coordinates)
    normal_sfm = np.zeros((count, 3), dtype=np.float64)
    normal_laser = np.zeros((count, 3), dtype=np.float64)
    support_counts = np.zeros(count, dtype=np.int64)
    keep = np.zeros(count, dtype=bool)
    if len(support_xy) < MIN_NEIGHBORS:
        return normal_laser, normal_sfm, keep, support_counts
    distances, indices = cKDTree(support_xy).query(coordinates, k=MAX_NEIGHBORS,
                                                 distance_upper_bound=RADIUS)
    for row in range(count):
        selected = indices[row, np.isfinite(distances[row])]
        support_counts[row] = len(selected)
        if len(selected) < MIN_NEIGHBORS:
            continue
        sfm, laser = plane_normal(sfm_cloud[selected]), plane_normal(laser_cloud[selected])
        if sfm is not None and laser is not None:
            normal_sfm[row], normal_laser[row], keep[row] = sfm, laser, True
    return normal_laser, normal_sfm, keep, support_counts


def tracked_cloud(reconstruction, image):
    # Use only points observed by this image. De-duplicate point IDs before kNN.
    point_ids = sorted({p.point3D_id for p in image.points2D if p.has_point3D()})
    world = np.asarray([reconstruction.points3D[i].xyz for i in point_ids], dtype=np.float64)
    if not len(world):
        return np.empty((0, 3))
    return np.asarray(image.cam_from_world() * world)


def freeze_protocol(args):
    path = DATA / "manifests/eth3d/annotation_sensitivity_protocol.json"
    split_path = DATA / "manifests/eth3d/frozen_scene_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    scene_inputs = {}
    for label in ("train", "validation", "test"):
        for scene in split[label]:
            folder = args.root / scene / "dslr_calibration_undistorted"
            scene_inputs[scene] = {name: sha256_file(folder / name)
                                  for name in ("cameras.txt", "images.txt", "points3D.txt")}
    scientific = {
        "protocol_id": PROTOCOL_ID, "runner_sha256": sha256_file(Path(__file__)),
        "core_source_sha256": source_sha256(), "scene_input_sha256": scene_inputs,
        "split_sha256": sha256_file(split_path), "split": split,
        "depth_manifest_sha256": sha256_file(DATA / "manifests/eth3d/depth_preprocessing/all_images.json"),
        "settings": {"radius_native_pixels": RADIUS, "maximum_neighbors": MAX_NEIGHBORS,
                     "minimum_neighbors": MIN_NEIGHBORS, "max_smallest_to_middle_eigenvalue": MAX_EIGEN_RATIO},
        "support": "Only image-observed unique SfM tracks with positive projected z and finite positive nearest-pixel laser depth; identical projected subpixel rays in both clouds; no source-agreement filter",
        "sampling": "Restrict previously frozen <=128 native features/image to common local support; no replacement, no new feature extraction or scene reassignment",
        "normal": "Same unweighted local TLS plane estimator for both sources; unit length, camera-z positive; both must pass identical planarity/rank checks",
        "analysis": "Separate train-only probes per label source; ridge selected on same frozen validation scenes; original default grid retained; equal-weight 5 test-scene paired NSI differences with 10000 bootstrap replicates; no probe-refit uncertainty",
        "scope": "Exploratory after default ETH3D results, frozen before these source comparisons; sparse common-support source sensitivity, not dense-MVS or independent camera-calibration validation",
        "primary_format_reference": "https://colmap.github.io/format.html",
        "models": [i for i, _ in MODELS],
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["scientific_protocol"] != scientific:
            raise ValueError("Frozen annotation protocol changed; use a distinct study version")
    else:
        write_json(path, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                          "scientific_protocol": scientific})
    return path, split


def export_model(identifier, args, protocol, split, depth_records):
    source = DATA / "manifests/features" / ("eth3d_" + identifier + ".npz")
    base_id = "eth3d_common_support_" + identifier
    paths = {label: DATA / "manifests/features" / (base_id + "_" + label + ".npz")
             for label in ("laser", "sfm")}
    report_path = RESULTS / "eth3d_annotation_sensitivity" / (base_id + ".build.json")
    input_sha, protocol_sha = sha256_file(source), sha256_file(protocol)
    if report_path.exists() and all(p.exists() for p in paths.values()):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["input_sha256"] != input_sha or report["protocol_sha256"] != protocol_sha:
            raise ValueError("Annotation export provenance changed")
        for label, path in paths.items():
            if sha256_file(path) != report["outputs"][label]["sha256"]:
                raise ValueError("Annotation export changed")
        return paths
    dataset = load_feature_npz(source)
    parsed = [str(value).split(":") for value in dataset.sample_ids]
    rows_by_scene_image = {}
    for index, (scene, image, native_index) in enumerate(parsed):
        rows_by_scene_image.setdefault((scene, image), []).append((index, int(native_index)))
    sfm_normals = np.zeros_like(dataset.geometry)
    laser_normals = np.zeros_like(dataset.geometry)
    retained = np.zeros(len(dataset.features), dtype=bool)
    coverage = []
    support_distribution = Counter()
    for label in ("train", "validation", "test"):
        for scene in split[label]:
            reconstruction = pycolmap.Reconstruction(str(args.root / scene / "dslr_calibration_undistorted"))
            image_by_name = {Path(image.name).name: image for image in reconstruction.images.values()}
            for (this_scene, name), rows in rows_by_scene_image.items():
                if this_scene != scene:
                    continue
                record = depth_records[scene, name]
                depth_path = Path(record["output"])
                # Full-depth bytes were verified during the default export. Verify
                # them again here once per image; cached normals never bypass SHA.
                if sha256_file(depth_path) != record["output_sha256"]:
                    raise ValueError("Calibrated depth changed")
                depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
                local_path = DATA / "cache/features" / ("eth3d_" + identifier) / scene / (name + ".npz")
                with np.load(local_path, allow_pickle=False) as local:
                    global_rows, native_rows = np.asarray(rows, dtype=np.int64).T
                    xy = local["coordinates_xy"][native_rows]
                    if not np.array_equal(local["descriptors"][native_rows].astype(np.float32),
                                          dataset.features[global_rows].astype(np.float32)):
                        raise ValueError("Native descriptor cache and frozen sample IDs differ")
                image = image_by_name[name]
                cloud = tracked_cloud(reconstruction, image)
                support_xy, cloud_sfm, cloud_laser = common_clouds(cloud, image.camera, depth)
                ng, ns, valid, counts = paired_normals(xy, support_xy, cloud_sfm, cloud_laser)
                retained[global_rows[valid]] = True
                laser_normals[global_rows[valid]] = ng[valid]
                sfm_normals[global_rows[valid]] = ns[valid]
                support_distribution.update(int(v) for v in counts[valid])
                coverage.append({"scene": scene, "image": name, "split": label,
                                 "original_samples": len(rows), "common_samples": int(valid.sum()),
                                 "image_tracked_points": len(cloud), "common_laser_tracked_points": len(support_xy),
                                 "native_cache_sha256": sha256_file(local_path)})
                del depth
            print(f"Annotation export {identifier}: {scene}; common={int(retained.sum())}", flush=True)
            del reconstruction
    common_arrays = {"features": dataset.features[retained].astype(np.float32),
                     "appearance": dataset.appearance[retained].astype(np.float32),
                     "groups": dataset.groups[retained], "split": dataset.split[retained],
                     "sample_ids": dataset.sample_ids[retained]}
    group_counts = {scene: int(np.sum(common_arrays["groups"] == scene))
                    for label in ("train", "validation", "test") for scene in split[label]}
    if any(value < 12 for value in group_counts.values()):
        write_json(report_path.with_suffix(".ineligible.json"), {"status": "INSUFFICIENT_COMMON_SUPPORT",
                   "input_sha256": input_sha, "protocol_sha256": protocol_sha,
                   "group_counts": group_counts, "coverage": coverage})
        raise ValueError(f"At least one frozen scene has fewer than 12 common samples: {group_counts}")
    outputs = {}
    for label, normals in (("laser", laser_normals), ("sfm", sfm_normals)):
        metadata = {**dataset.metadata, "protocol": PROTOCOL_ID, "source_feature_sha256": input_sha,
                    "annotation_source": label + " common-support local plane normals",
                    "protocol_sha256": protocol_sha, "original_normal_operator": "replaced by matched-support TLS planes"}
        arrays = {**common_arrays, "geometry": normals[retained].astype(np.float32)}
        FeatureDataset(**arrays, metadata=metadata).validate()
        _atomic_savez(paths[label], compressed=True, **arrays,
                      metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        outputs[label] = {"path": str(paths[label]), "sha256": sha256_file(paths[label])}
    dot = np.sum(laser_normals[retained] * sfm_normals[retained], axis=1)
    angles = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    report = {"status": "PASS", "input_sha256": input_sha, "protocol_sha256": protocol_sha,
              "original_sample_count": len(retained), "common_sample_count": int(retained.sum()),
              "group_counts": group_counts, "coverage": coverage, "support_count_distribution": support_distribution,
              "axial_normal_angle_degrees": {"median": float(np.median(angles)), "mean": float(np.mean(angles)),
                                            "p95": float(np.percentile(angles, 95))},
              "outputs": outputs, "runner_sha256": sha256_file(Path(__file__))}
    write_json(report_path, report)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / ".cache/eth3d_training")
    parser.add_argument("--models", nargs="*", choices=[i for i, _ in MODELS])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol, split = freeze_protocol(args)
    records = json.loads((DATA / "manifests/eth3d/depth_preprocessing/all_images.json").read_text(encoding="utf-8"))
    records = {(r["scene"], Path(r["image"]).name): r for r in records if r["state"] == "prepared"}
    config = load_config(ROOT / "configs/default.toml")
    folder = RESULTS / "eth3d_annotation_sensitivity"
    folder.mkdir(parents=True, exist_ok=True)
    for identifier, _ in MODELS:
        if args.models and identifier not in args.models:
            continue
        paths = export_model(identifier, args, protocol, split, records)
        if not args.prepare_only:
            for label, path in paths.items():
                output = folder / (identifier + "_" + label + ".json")
                if output.exists():
                    old = json.loads(output.read_text(encoding="utf-8"))
                    if old["provenance"]["input_sha256"] != sha256_file(path):
                        raise ValueError("Paired NSI result input changed")
                else:
                    p = compute_nsi_artifacts(path, output, config)
                    print(f"Annotation NSI {identifier} {label}: {p['result']['NSI']:.8g}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
