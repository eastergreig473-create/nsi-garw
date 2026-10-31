"""Create calibrated pinhole depth caches only after archive CRC checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pycolmap

from nsi_garw.eth3d import read_cameras, read_images, read_raw_depth, depth_on_pinhole
from nsi_garw.io import sha256_file
import extract_eth3d


def wait_for(path: Path, wait: bool):
    deadline = time.monotonic() + 8 * 3600
    while not path.exists():
        if not wait or time.monotonic() > deadline:
            raise FileNotFoundError(path)
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checks = Path("F:/NSI_GARW_Data/manifests/eth3d/archive_checks") / hashlib.sha256(str(root).encode()).hexdigest()[:12]
    jpg = Path("F:/NSI_GARW_Data/downloads/ETH3D_openssl/multi_view_training_dslr_jpg.7z")
    wait_for(jpg, args.wait)
    # Preserve the F: archive; verify/extract the original camera calibration.
    extract_eth3d.OUTPUT, extract_eth3d.AUDIT = root, checks
    extract_eth3d.extract(jpg)
    split_path = Path("F:/NSI_GARW_Data/manifests/eth3d/frozen_scene_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    destination = args.output.resolve()
    manifest_root = Path("F:/NSI_GARW_Data/manifests/eth3d/depth_preprocessing")
    manifest_root.mkdir(parents=True, exist_ok=True)
    records = []
    for label in ("train", "validation", "test"):
        for scene in split[label]:
            wait_for(checks / "multi_view_training_dslr_undistorted.7z.json", args.wait)
            wait_for(checks / f"{scene}_dslr_depth.7z.json", args.wait)
            folder = root / scene
            original_path = folder / "dslr_calibration_jpg"
            target_path = folder / "dslr_calibration_undistorted"
            original_cameras = read_cameras(original_path / "cameras.txt")
            target_cameras = read_cameras(target_path / "cameras.txt")
            original_images = read_images(original_path / "images.txt")
            target_images = read_images(target_path / "images.txt")
            for name, target_image in sorted(target_images.items()):
                original_image = original_images[name]
                first = original_cameras[original_image["camera_id"]]
                second = target_cameras[target_image["camera_id"]]
                depth_path = folder / "ground_truth_depth/dslr_images" / name
                image_path = folder / "images" / target_image["name"]
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    if image.size != (second.width, second.height):
                        raise ValueError("Undistorted image dimensions differ from calibration")
                output = destination / scene / (name + ".npy")
                record_path = manifest_root / (scene + "_" + name + ".json")
                if record_path.exists():
                    records.append(json.loads(record_path.read_text(encoding="utf-8")))
                    continue
                record = {"scene": scene, "split": label, "image": str(image_path), "raw_depth": str(depth_path),
                    "raw_camera": str(first), "pinhole_camera": str(second), "pycolmap_version": pycolmap.__version__,
                    "split_manifest_sha256": sha256_file(split_path),
                    "original_cameras_sha256": sha256_file(original_path / "cameras.txt"),
                    "pinhole_cameras_sha256": sha256_file(target_path / "cameras.txt"),
                    "mapping": "pycolmap exact inverse camera map; top-left corner origin; centers at index+0.5; nearest depth; outside/missing remain NaN"}
                if not depth_path.exists():
                    record.update({"state": "no_public_depth_for_image", "output": None})
                else:
                    raw = read_raw_depth(depth_path, first.width, first.height)
                    depth = depth_on_pinhole(raw, first, second)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(".writing.npy")
                    np.save(temporary, depth, allow_pickle=False)
                    temporary.replace(output)
                    record.update({"state": "prepared", "output": str(output), "output_sha256": sha256_file(output),
                        "raw_depth_sha256": sha256_file(depth_path), "image_sha256": sha256_file(image_path),
                        "valid_depth_fraction": float(np.isfinite(depth).mean())})
                    del raw, depth
                record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                records.append(record)
                print(f"ETH3D calibrated depth: {scene}/{name} {record['state']}", flush=True)
    (manifest_root / "all_images.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared ETH3D manifest: {len(records)} image records", flush=True)


if __name__ == "__main__":
    main()
