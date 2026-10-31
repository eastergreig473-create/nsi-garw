"""Prepare and run pinned COLMAP MVS using supplied ETH3D calibrated poses.

CPU preparation may overlap the learned-texture run. CUDA PatchMatch starts
only after that GPU job has exited and all 45 texture NSI results exist.
The prepared and reconstructed maps are not scientific sensitivity results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
import pycolmap
from scipy.spatial.transform import Rotation

from nsi_garw.experiments import write_json
from nsi_garw.io import sha256_file
from run_hpatches_suite import DATA, MODELS, RESULTS, ROOT

VERSION = "3.9.1"
IMAGE_SIZE = 1600
SOURCES = 10


def installation():
    path = DATA / f"manifests/colmap/installation_{VERSION}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    exe = next(Path(m["path"]) for m in record["members"] if Path(m["path"]).name.lower() == "colmap.exe")
    member = next(m for m in record["members"] if Path(m["path"]) == exe)
    if record["status"] != "PASS" or sha256_file(exe) != member["sha256"]:
        raise ValueError("Pinned executable verification failed")
    return path, exe


def invoke(exe, arguments, logfile):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PATH"] = os.pathsep.join([str(exe.parent), str(exe.parent.parent / "lib"), environment.get("PATH", "")])
    environment["QT_PLUGIN_PATH"] = str(exe.parent.parent / "lib/plugins")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as stream:
        process = subprocess.run([str(exe), *[str(a) for a in arguments]], env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"COLMAP failed ({process.returncode}); see {logfile}")


def check_prepared(original, workspace):
    source, target = pycolmap.Reconstruction(str(original)), pycolmap.Reconstruction(str(workspace / "sparse"))
    source_images = {i.name: i for i in source.images.values()}
    if set(source_images) != {i.name for i in target.images.values()}:
        raise ValueError("MVS preparation changed image inventory")
    records = []
    for image in target.images.values():
        previous = source_images[image.name]
        raw_pose_error = float(np.max(np.abs(image.cam_from_world().matrix()-previous.cam_from_world().matrix())))
        # ETH3D text quaternions are rounded, hence not exactly unit length.
        # Legacy undistortion normalizes them. Compare rigid rotations after
        # the same unit-quaternion normalization instead of raw matrix noise.
        def normalized_pose(item):
            pose = item.cam_from_world()
            return np.column_stack([Rotation.from_quat(pose.rotation.quat).as_matrix(), pose.translation])
        pose_error = float(np.max(np.abs(normalized_pose(image)-normalized_pose(previous))))
        if pose_error > 1e-10 or image.camera.model_name != "PINHOLE":
            raise ValueError("MVS preparation changed poses or pinhole camera model")
        path = workspace / "images" / image.name
        if not path.exists():
            raise FileNotFoundError(path)
        records.append({"image": image.name, "width": image.camera.width, "height": image.camera.height,
                        "camera_params": image.camera.params.tolist(), "source_camera_params": previous.camera.params.tolist(),
                        "image_sha256": sha256_file(path), "pose_max_error": pose_error,
                        "raw_matrix_error_before_unit_quaternion_normalization": raw_pose_error})
    return records


def wait_gpu_queue():
    expected = [RESULTS / "learned_texture_nsi" / f"{identifier}_texturegrid_s{level:03d}.json"
                for identifier, _ in MODELS if identifier != "sift_default" for level in (20, 40, 60, 80, 100)]
    print("MVS GPU queue waits for 45/45 learned-texture results and texture runner exit", flush=True)
    deadline = time.monotonic()+24*3600
    while True:
        running = False
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                if any("run_deep_texture_nsi.py" in token for token in (process.info["cmdline"] or [])):
                    running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if all(p.exists() for p in expected) and not running:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Texture GPU queue did not finish in 24h")
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-texture", action="store_true")
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    install_path, exe = installation()
    split_path = DATA / "manifests/eth3d/frozen_scene_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    scene_names = [scene for label in ("train", "validation", "test") for scene in split[label]]
    if args.scenes:
        if not set(args.scenes).issubset(scene_names):
            raise ValueError("Unknown scene requested")
        scene_names = [s for s in scene_names if s in args.scenes]
    # An ASCII junction avoids legacy Windows C++ narrow-path handling of
    # the Chinese project path. Never replace/delete an existing directory.
    source_root = Path("D:/NSI_GARW_ETH3D")
    if source_root.resolve() != (ROOT / ".cache/eth3d_training").resolve():
        raise ValueError("ASCII ETH3D junction does not point to the verified data")
    output_root = DATA / "cache/eth3d_colmap_dense_1600"
    protocol_path = DATA / "manifests/eth3d/colmap_dense_protocol_v2.json"
    records = json.loads((DATA / "manifests/eth3d/depth_preprocessing/all_images.json").read_text(encoding="utf-8"))
    settings = {
        "version": VERSION, "executable_sha256": sha256_file(exe), "installation_sha256": sha256_file(install_path),
        "runner_sha256": sha256_file(Path(__file__)), "split_sha256": sha256_file(split_path),
        "candidate_scenes": [scene for label in ("train", "validation", "test") for scene in split[label]],
        "max_image_size": IMAGE_SIZE, "source_views": SOURCES, "seed": 20261023,
        "patch_match": {"geom_consistency": True, "num_iterations": 5, "window_radius": 5,
                        "window_step": 1, "num_samples": 15, "filter": True,
                        "filter_min_num_consistent": 2, "gpu_index": "0", "cache_size_gb": 8},
        "calibration_inputs": {scene: {name: sha256_file(source_root / scene / "dslr_calibration_undistorted" / name)
                                       for name in ("cameras.txt", "images.txt", "points3D.txt")}
                               for label in ("train", "validation", "test") for scene in split[label]},
        "scope": "Exploratory source sensitivity preparation; supplied fixed poses/intrinsics and sparse points choose views/depth range; MVS uses RGB only, never laser depth or NSI outcomes",
        "version_choice": "4.2.1 portable build uses CUDA13.2; installed driver545.84 supports older runtime. Pin official3.9.1 portable CUDA instead of changing driver during active research jobs.",
        "pose_check": "Equal translations and equal rigid rotations after normalizing rounded ETH3D text quaternions, tolerance1e-10; raw matrix differences separately retained. V1 CPU preparation stopped at overly strict unnormalized quaternion check before any CUDA MVS/NSI outcome.",
        "primary_method": "https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses",
    }
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8"))["scientific_settings"] != settings:
            raise ValueError("Frozen dense protocol changed")
    else:
        write_json(protocol_path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scientific_settings": settings})
    audit_root = DATA / "manifests/eth3d/colmap_dense"
    audit_root.mkdir(parents=True, exist_ok=True)
    for scene in scene_names:
        workspace = output_root / scene
        prepared = audit_root / (scene+"_prepared.json")
        if prepared.exists():
            old = json.loads(prepared.read_text(encoding="utf-8"))
            if old["protocol_sha256"] != sha256_file(protocol_path):
                raise ValueError("Prepared workspace protocol differs")
            continue
        for record in records:
            if record["scene"] == scene and sha256_file(record["image"]) != record["image_sha256"]:
                raise ValueError("Original input image SHA differs")
        workspace.mkdir(parents=True, exist_ok=True)
        invoke(exe, ["image_undistorter", "--image_path", source_root / scene / "images",
                     "--input_path", source_root / scene / "dslr_calibration_undistorted",
                     "--output_path", workspace, "--output_type", "COLMAP", "--max_image_size", IMAGE_SIZE,
                     "--num_patch_match_src_images", SOURCES], audit_root / (scene+"_prepare.log"))
        images = check_prepared(source_root / scene / "dslr_calibration_undistorted", workspace)
        write_json(prepared, {"status": "PASS", "protocol_sha256": sha256_file(protocol_path),
                             "image_count": len(images), "images": images,
                             "patch_match_config_sha256": sha256_file(workspace / "stereo/patch-match.cfg")})
        print(f"MVS prepared {scene}: {len(images)} images; poses unchanged", flush=True)
    if args.prepare_only:
        return
    if args.wait_for_texture:
        wait_gpu_queue()
    for scene in scene_names:
        workspace = output_root / scene
        complete = audit_root / (scene+"_complete.json")
        if complete.exists():
            continue
        print(f"MVS CUDA START {scene}", flush=True)
        started = time.perf_counter()
        invoke(exe, ["patch_match_stereo", "--workspace_path", workspace, "--workspace_format", "COLMAP",
                     "--random_seed", 20261023, "--PatchMatchStereo.gpu_index", "0",
                     "--PatchMatchStereo.max_image_size", IMAGE_SIZE, "--PatchMatchStereo.cache_size", 8,
                     "--PatchMatchStereo.geom_consistency", "true"], audit_root / (scene+"_patch_match.log"))
        prepared = json.loads((audit_root / (scene+"_prepared.json")).read_text(encoding="utf-8"))
        outputs = []
        for image in prepared["images"]:
            for kind in ("depth_maps", "normal_maps"):
                path = workspace / "stereo" / kind / (image["image"]+".geometric.bin")
                if not path.exists():
                    raise ValueError(f"MVS exited without expected geometric map: {path}")
                outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        write_json(complete, {"status": "PASS", "protocol_sha256": sha256_file(protocol_path),
                             "image_count": prepared["image_count"], "wall_seconds": time.perf_counter()-started,
                             "outputs": outputs, "scope": "Map production only; paired NSI comparison not yet computed"})
        print(f"MVS CUDA DONE {scene}: verified {len(outputs)} maps", flush=True)


if __name__ == "__main__":
    main()
