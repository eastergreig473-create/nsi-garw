"""GPU-serialized layer diagnostics after the cross-model feature queue."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from run_hpatches_suite import DATA, FEATURES, MODELS, RESULTS, ROOT, run_command
from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts
from nsi_garw.io import load_feature_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-cross-model", action="store_true")
    args = parser.parse_args()
    if args.wait_for_cross_model:
        deadline = time.monotonic() + 8 * 3600
        while not all((FEATURES / (name + ".npz")).exists() for name, _ in MODELS):
            if time.monotonic() > deadline:
                raise TimeoutError("Cross-model GPU queue has not completed")
            time.sleep(20)
    layers = []
    for layer in ("maxpool", "layer1", "layer2", "layer3", "layer4"):
        layers.append((f"resnet50_{layer}_crop224", ["--adapter", "resnet50", "--checkpoint",
            "weights/resnet50-11ad3fa6.pth", "--layer", layer]))
    for layer in (2, 5, 8):
        layers.append((f"dinov2_vits14_block{layer}_518", ["--adapter", "dinov2", "--repository",
            "external/dinov2", "--checkpoint", str(DATA / "models/dinov2_vits14_pretrain.pth"),
            "--layer", str(layer), "--image-size", "518"]))
    output = RESULTS / "layerwise"
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/default.toml")
    for identifier, adapter in layers:
        features = FEATURES / (identifier + ".npz")
        if not features.exists():
            run_command(identifier, "layer_features", ["build-hpatches-model", *adapter,
                "--export-id", identifier, "--pairs-manifest", str(DATA / "manifests/hpatches/hpatches_pairs.csv"),
                "--dataset-root", str(DATA / "datasets/HPatches/hpatches-sequences-release"),
                "--cache-root", str(DATA / "cache"), "--output", str(features),
                "--max-samples-per-pair", "64", "--seed", "20261023", "--expected-pairs", "580",
                "--proxy", "ncc", "--device", "cuda"])
        result = output / (identifier + ".json")
        if not result.exists():
            compute_nsi_artifacts(features, result, config)
        print(f"Layer complete: {identifier}", flush=True)
    rows = []
    targets = [(name, output / (name + ".json")) for name, _ in layers]
    targets.append(("dinov2_vits14_block11_518", RESULTS / "dinov2_vits14_block11_518.json"))
    for identifier, path in targets:
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = load_feature_npz(FEATURES / (identifier + ".npz"))
        rows.append({"experiment_id": identifier, "model": data.metadata["model"],
            "layer": data.metadata["layer"], "samples": len(data.features),
            "dimension": data.features.shape[1], **payload["result"],
            "result_json": str(path)})
    with (output / "layerwise_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
