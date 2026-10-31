"""Complete paired LoFTR/RoMa reference benchmarks, separate from encoder reranking."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as functional

from nsi_garw.adapters import LoFTRBackboneAdapter
from nsi_garw.degradations import apply_degradation
from nsi_garw.evaluation import homography_reprojection_errors
from nsi_garw.io import sha256_file
from nsi_garw.matching_builders import summarize_pair_matching
from run_hpatches_suite import DATA, ROOT, RESULTS


def resized_to_original(points, original_hw, resized_hw):
    original_h, original_w = original_hw
    resized_h, resized_w = resized_hw
    return (np.asarray(points) + .5) * np.array([original_w/resized_w, original_h/resized_h]) - .5


def loftr_image(image):
    original = image.shape[:2]
    resized = LoFTRBackboneAdapter._resized_shape(original, 1024)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tensor = torch.from_numpy(gray.copy())[None, None].to(device="cuda", dtype=torch.float32)/255.
    return functional.interpolate(tensor, size=resized, mode="bilinear", align_corners=False, antialias=True), resized


def create_model(name):
    if name == "loftr":
        adapter = LoFTRBackboneAdapter(DATA / "models/loftr_outdoor.ckpt", device="cuda")
        return adapter.model
    repository = ROOT / "external/roma"
    sys.path.insert(0, str(repository))
    from romatch import roma_outdoor
    model = roma_outdoor(device="cpu",
        weights=torch.load(DATA / "models/roma_outdoor.pth", map_location="cpu", weights_only=True),
        dinov2_weights=torch.load(DATA / "models/dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True),
        coarse_res=560, upsample_res=560, amp_dtype=torch.float16,
        symmetric=False, use_custom_corr=False, upsample_preds=False)
    model.to("cuda")
    model.encoder.amp = True
    model.encoder.amp_dtype = torch.float16
    model.encoder.dinov2_vitl14[0] = model.encoder.dinov2_vitl14[0].to(device="cuda", dtype=torch.float16).eval()
    return model


def infer(model, name, first, second):
    with torch.inference_mode():
        if name == "loftr":
            x, shape_x = loftr_image(first)
            y, shape_y = loftr_image(second)
            prediction = model({"image0": x, "image1": y})
            a = resized_to_original(prediction["keypoints0"].float().cpu().numpy(), first.shape[:2], shape_x)
            b = resized_to_original(prediction["keypoints1"].float().cpu().numpy(), second.shape[:2], shape_y)
            score = prediction["confidence"].float().cpu().numpy()
        else:
            warp, certainty = model.match(Image.fromarray(first), Image.fromarray(second), device="cuda")
            if torch.isfinite(certainty).all() and float(certainty.clamp_min(0).sum()) > 0:
                matches, confidence = model.sample(warp, certainty, num=5000)
                a, b = model.to_pixel_coordinates(matches, *first.shape[:2], *second.shape[:2])
                # Upstream returns top-left-corner coordinates; homographies use zero-based pixel centers.
                a, b = a.float().cpu().numpy()-.5, b.float().cpu().numpy()-.5
                score = confidence.float().cpu().numpy()
            else:
                a, b, score = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return a, b, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", choices=["loftr", "roma"], default=["loftr", "roma"])
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    pairs_path = DATA / "manifests/hpatches/hpatches_pairs.csv"
    with pairs_path.open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    image_root = DATA / "datasets/HPatches/hpatches-sequences-release"
    runner_sha = sha256_file(Path(__file__))
    for name in args.models:
        print(f"Native matcher START: {name}", flush=True)
        model = create_model(name)
        first_row = pairs[0]
        first = np.asarray(Image.open(image_root / first_row["reference_image"]).convert("RGB"))
        second = np.asarray(Image.open(image_root / first_row["target_image"]).convert("RGB"))
        for warmup in range(3):
            torch.manual_seed(20261023+warmup)
            infer(model, name, first, second)
        for condition, strength in (("clean", 0.), ("texture", .8)):
            folder = RESULTS / "native_matchers" / f"{name}_{condition}_s{round(strength*100):03d}"
            folder.mkdir(parents=True, exist_ok=True)
            records = []
            for index, row in enumerate(pairs):
                item = folder / "per_pair" / (row["pair_id"] + ".json")
                if item.exists():
                    record = json.loads(item.read_text(encoding="utf-8"))
                    if record["runner_sha256"] != runner_sha:
                        raise ValueError("Existing native-matcher result has another runner version")
                    records.append(record)
                    continue
                first = np.asarray(Image.open(image_root / row["reference_image"]).convert("RGB"))
                second = np.asarray(Image.open(image_root / row["target_image"]).convert("RGB"))
                pair_seed = int.from_bytes(hashlib.sha256(f"20261023:{row['pair_id']}:{condition}".encode()).digest()[:8], "little")
                if condition != "clean":
                    second = np.clip(np.rint(apply_degradation(second, condition, strength, seed=pair_seed)*255.), 0, 255).astype(np.uint8)
                torch.manual_seed(pair_seed % (2**63-1))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                a, b, confidence = infer(model, name, first, second)
                torch.cuda.synchronize()
                elapsed, peak = time.perf_counter()-started, torch.cuda.max_memory_allocated()
                errors = homography_reprojection_errors(a, b, np.loadtxt(image_root / row["homography"]), symmetric=True)
                finite, count = np.isfinite(errors), len(errors)
                record = {"pair_id": row["pair_id"], "scene": row["sequence"], "split": row["split"], "match_count": count,
                    "invalid_projection_count": int((~finite).sum()), "matcher_seconds": elapsed, "cuda_peak_allocated_bytes": peak,
                    "runner_sha256": runner_sha, "seed": pair_seed}
                for threshold in (1., 3., 5., 10.):
                    record[f"mma_at_{threshold:g}"] = float(np.sum(finite & (errors <= threshold))/count) if count else 0.
                item.parent.mkdir(exist_ok=True)
                item.write_text(json.dumps(record, indent=2), encoding="utf-8")
                cache = DATA / "cache/native_matches" / folder.name
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache / (row["pair_id"] + ".npz"), first_xy=a, second_xy=b, confidence=confidence)
                records.append(record)
                if (index+1) % 10 == 0 or index == 0:
                    print(f"Native {name} {condition}: {index+1}/{len(pairs)}; MMA@3={record['mma_at_3']:.4f}; matches={count}", flush=True)
            pd.DataFrame(records).to_csv(folder / "per_pair.csv", index=False)
            payload = {"model": name, "condition": condition, "strength": strength,
                "scope": "complete paired native matcher reference; separate candidate sets and metric from encoder top1 reranking",
                "preprocessing": "LoFTR: OpenCV grayscale, long edge1024, multiple-of8, bilinear antialias, FP32; RoMa: official RGB ImageNet preprocessing,560 square, CUDA AMP, one-way,no high-resolution upsampling,5000 balanced native samples",
                "coordinates": "zero-based original-image pixel centers; half-pixel correction applied to both resize and RoMa normalized conversion",
                "error": "RMS symmetric homography error, thresholds1/3/5/10; zero-match pairs retained with zero MMA",
                "timing": "three warmups; synchronized matcher preprocessing/inference/readout; image IO and synthetic corruption excluded",
                "pairs_manifest_sha256": sha256_file(pairs_path), "runner_sha256": runner_sha,
                "results": summarize_pair_matching(records), "processed_pair_count": len(records)}
            (folder / "matching_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Native matcher DONE {name} {condition}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
