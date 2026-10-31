"""Held-out GARW candidate reranking, including unretrieved positives."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates

from nsi_garw.attributes import sobel_magnitude
from nsi_garw.cli import _batch_adapter, build_parser
from nsi_garw.config import load_config
from nsi_garw.degradations import apply_degradation
from nsi_garw.evaluation import homography_reprojection_errors
from nsi_garw.experiments import run_garw_table, source_sha256
from nsi_garw.feature_builders import _load_local_features, _save_local_features, _deterministic_subset
from nsi_garw.garw import standardized_projected_cosine
from nsi_garw.io import load_feature_npz, sha256_file
from nsi_garw.local_proxy import fit_local_nsi_proxy
from nsi_garw.metrics import (fit_ridge_probe, orthonormal_basis, normalized_subspace_overlap,
    residualized_appearance_basis, estimate_information_purity, r2_per_output, mean_finite)
from run_hpatches_suite import DATA, MODELS, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-id", default="loftr_outdoor_coarse_1024")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--queries-per-pair", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--kind", choices=["clean", "texture", "illumination", "motion_blur", "jpeg", "repetitive", "occlusion", "composite"], default="clean")
    parser.add_argument("--strength", type=float, default=0.)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--fresh-reference-cache", action="store_true")
    args = parser.parse_args()
    if args.run_tag and not re.fullmatch(r"[a-z0-9_]+", args.run_tag):
        raise ValueError("Run tag must contain only lowercase letters, digits, or underscores")
    if args.fresh_reference_cache and not args.run_tag:
        raise ValueError("A distinct run tag is required to preserve the previous reference cache")
    cv2.setNumThreads(2)
    identifier = args.export_id
    if args.kind != "clean" and identifier not in {"loftr_outdoor_coarse_1024", "roma_outdoor_scale16_560"}:
        raise ValueError("Degraded GARW currently requires a spatially stable LoFTR/RoMa encoder grid")
    suffix = "" if args.kind == "clean" else f"_{args.kind}_s{round(args.strength * 100):03d}"
    run_suffix = suffix + ("_" + args.run_tag if args.run_tag else "")
    adapter_args = dict(MODELS)[identifier]
    pairs_path = DATA / "manifests/hpatches/hpatches_pairs.csv"
    feature_path = DATA / "manifests/features" / f"{identifier}.npz"
    config = load_config(Path("configs/default.toml"))
    frozen_hash = source_sha256()
    reference_preprocessing = set()
    target_preprocessing = set()
    with pairs_path.open(encoding="utf-8", newline="") as handle:
        pairs = [r for r in csv.DictReader(handle) if r["split"] in {"validation", "test"}]
    dataset = load_feature_npz(feature_path)
    result = json.loads((RESULTS / f"{identifier}.json").read_text(encoding="utf-8"))["result"]
    train = dataset.split == "train"
    # Keep calibration and test sequences outside the probes used by GARW.
    gp = fit_ridge_probe(dataset.features[train], dataset.geometry[train], result["geometry_alpha"])
    ap = fit_ridge_probe(dataset.features[train], dataset.appearance[train], result["appearance_alpha"])
    ug = orthonormal_basis(gp.coefficients_standardized, 8, 1e-6)
    ua_raw = orthonormal_basis(ap.coefficients_standardized, 8, 1e-6)
    ua = residualized_appearance_basis(ug, ap.coefficients_standardized, 8, 1e-6)
    separation = 1. - normalized_subspace_overlap(ug, ua_raw)
    pair_labels = np.asarray([str(s).rsplit(":", 1)[0] for s in dataset.sample_ids])
    output = RESULTS / "garw_hpatches" / (identifier + run_suffix)
    output.mkdir(parents=True, exist_ok=True)
    candidate_file = DATA / "manifests/garw" / f"{identifier}{run_suffix}_q{args.queries_per_pair}_k{args.candidates}.csv"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    if not candidate_file.exists():
        arguments = build_parser().parse_args(["build-hpatches-model", *adapter_args,
            "--export-id", identifier, "--pairs-manifest", str(pairs_path),
            "--dataset-root", str(DATA / "datasets/HPatches/hpatches-sequences-release"),
            "--cache-root", str(DATA / "cache"), "--output", str(DATA / "cache/unused.npz"),
            "--device", args.device])
        adapter = _batch_adapter(arguments)
        records = []
        for pi, row in enumerate(pairs):
            image_root = DATA / "datasets/HPatches/hpatches-sequences-release"
            reference_image = np.asarray(Image.open(image_root / row["reference_image"]).convert("RGB"))
            reference_identifier = identifier + ("_reference_" + args.run_tag if args.fresh_reference_cache else "")
            cache = DATA / "cache/features" / reference_identifier / row["sequence"]
            reference_path = cache / "1.npz"
            if reference_path.exists():
                first = _load_local_features(reference_path)
            else:
                first = adapter.extract(reference_image)
                _save_local_features(reference_path, first)
            target_cache = DATA / "cache/features" / (identifier + suffix) / row["sequence"] / (Path(row["target_image"]).stem + ".npz")
            if target_cache.exists():
                second = _load_local_features(target_cache)
            else:
                target_image = np.asarray(Image.open(image_root / row["target_image"]).convert("RGB"))
                if args.kind != "clean":
                    pair_seed = int.from_bytes(hashlib.sha256(f"20261023:{row['pair_id']}:{args.kind}".encode()).digest()[:8], "little")
                    target_image = np.clip(np.rint(apply_degradation(target_image, args.kind, args.strength, seed=pair_seed) * 255.), 0, 255).astype(np.uint8)
                second = adapter.extract(target_image)
                if args.kind != "clean":
                    second.metadata["preprocessing"] += f"; target-only {args.kind} strength {args.strength:g}"
                _save_local_features(target_cache, second)
            reference_preprocessing.add(first.metadata["preprocessing"])
            target_preprocessing.add(second.metadata["preprocessing"])
            selected = _deterministic_subset(len(first.descriptors), args.queries_per_pair,
                seed=config.seed, pair_id=row["pair_id"])
            x = np.asarray(first.descriptors[selected], dtype=np.float32)
            y = np.asarray(second.descriptors, dtype=np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            similarity = x @ y.T
            k = min(args.candidates, len(y))
            candidates = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
            query_indices = np.repeat(selected, k)
            target_indices = candidates.reshape(-1)
            sg = standardized_projected_cosine(first.descriptors[query_indices], second.descriptors[target_indices], ug, gp.x_mean, gp.x_scale)
            sa = standardized_projected_cosine(first.descriptors[query_indices], second.descriptors[target_indices], ua, gp.x_mean, gp.x_scale)
            h = np.loadtxt(image_root / row["homography"])
            errors = homography_reprojection_errors(first.coordinates_xy[query_indices], second.coordinates_xy[target_indices], h, symmetric=True)
            window = pair_labels == row["pair_id"]
            window_features = dataset.features[window]
            if args.kind != "clean":
                native_indices = np.asarray([int(str(s).rsplit(":", 1)[1]) for s in dataset.sample_ids[window]])
                window_features = second.descriptors[native_indices]
            pg, pa = gp.predict(window_features), ap.predict(window_features)
            accessible = .5 * (np.clip(mean_finite(r2_per_output(dataset.geometry[window], pg)), 0, 1)
                              + np.clip(mean_finite(r2_per_output(dataset.appearance[window], pa)), 0, 1))
            purity = estimate_information_purity(pg, pa, replace(config, seed=config.seed + pi + 1))
            reference_nsi = float(accessible * separation * purity.purity)
            texture = map_coordinates(sobel_magnitude(reference_image),
                [first.coordinates_xy[selected, 1], first.coordinates_xy[selected, 0]],
                order=1, mode="nearest")
            disagreement = np.abs(sg - sa).reshape(len(selected), k).mean(axis=1)
            ordered_scores = np.take_along_axis(similarity, candidates, axis=1)
            margin = ordered_scores[:, 0] - ordered_scores[:, min(1, k - 1)]
            for qi, source_index in enumerate(selected):
                for ci in range(k):
                    pos = qi * k + ci
                    records.append({"query_id": f"{row['pair_id']}:{int(source_index)}", "scene": row["sequence"],
                        "pair_id": row["pair_id"], "split": "calibration" if row["split"] == "validation" else "evaluation",
                        "label": int(np.isfinite(errors[pos]) and errors[pos] <= 3.),
                        "s_geometry": float(sg[pos]), "s_appearance": float(sa[pos]),
                        "s_baseline": float(ordered_scores[qi, ci]), "texture_energy": float(texture[qi]),
                        "projected_disagreement": float(disagreement[qi]), "raw_margin": float(margin[qi]),
                        "nsi_reference": reference_nsi, "reprojection_error": float(errors[pos]),
                        "candidate_rank_raw": ci + 1})
            if (pi + 1) % 10 == 0:
                print(f"GARW candidates {identifier}: {pi+1}/{len(pairs)} pairs", flush=True)
        table = pd.DataFrame(records)
        predicted, proxy, trace = fit_local_nsi_proxy(table,
            feature_columns=["texture_energy", "projected_disagreement", "raw_margin"],
            lambdas=config.ridge.lambdas, seed=config.seed)
        predicted.to_csv(candidate_file, index=False)
        (output / "proxy.json").write_text(json.dumps({"model": proxy.to_dict(), "trace": trace}, indent=2), encoding="utf-8")
        (output / "candidate_provenance.json").write_text(json.dumps({
            "source_sha256": frozen_hash, "input_sha256": sha256_file(feature_path),
            "runner_sha256": sha256_file(Path(__file__)), "run_tag": args.run_tag,
            "fresh_reference_cache": args.fresh_reference_cache, "reference_device": args.device,
            "pairs_manifest_sha256": sha256_file(pairs_path), "queries_per_pair": args.queries_per_pair,
            "candidate_count": args.candidates, "candidate_selection": "top-k normalized-descriptor similarity; ground truth used only for labels",
            "positive_rule": "RMS symmetric homography error <=3 pixels", "missing_positive_queries": "retained with zero AP MRR and top1",
            "proxy_target": "whole target-image window NSI from at most 64 aligned features; train-only fixed probes and subspace",
            "probe_training_sequences": 81, "calibration_sequences": 18, "evaluation_sequences": 17,
            "geometry_rank": ug.shape[1], "appearance_rank": ua.shape[1],
            "degradation_kind": args.kind, "degradation_strength": args.strength,
            "attribute_targets": "clean-image annotations retained when features are degraded; measure recovery of clean scene attributes",
            "reference_preprocessing": sorted(reference_preprocessing),
            "target_preprocessing": sorted(target_preprocessing),
            "scope": "encoder-feature candidate reranking; not the complete paired LoFTR/RoMa matcher"}, indent=2), encoding="utf-8")
    print(f"Calibrating GARW on {candidate_file}", flush=True)
    payload = run_garw_table(candidate_file, output / "non_oracle", config)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    oracle = pd.read_csv(candidate_file)
    oracle["nsi_local"], oracle["nsi_mode"] = oracle["nsi_reference"], "oracle_annotation"
    oracle_path = candidate_file.with_name(candidate_file.stem + "_oracle.csv")
    oracle.to_csv(oracle_path, index=False)
    run_garw_table(oracle_path, output / "oracle_annotation", config)


if __name__ == "__main__":
    main()
