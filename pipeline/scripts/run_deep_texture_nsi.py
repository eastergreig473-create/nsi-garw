"""Extend the frozen texture grid to all nine learned representations."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from nsi_garw.cli import _batch_adapter, build_parser
from nsi_garw.config import load_config
from nsi_garw.degradations import apply_degradation
from nsi_garw.experiments import compute_nsi_artifacts
from nsi_garw.feature_builders import build_hpatches_feature_dataset
from run_hpatches_suite import ROOT, DATA, MODELS, RESULTS


class TextureAdapter:
    def __init__(self, adapter, strength):
        self.adapter, self.strength = adapter, strength

    def extract(self, image):
        degraded = np.clip(np.rint(apply_degradation(image, "texture", self.strength, seed=20261023)*255.),0,255).astype(np.uint8)
        features = self.adapter.extract(degraded)
        features.metadata["preprocessing"] += f"; target-only texture suppression strength {self.strength:g}; RGB uint8 quantization"
        return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-native", action="store_true")
    args = parser.parse_args()
    if args.wait_for_native:
        expected = [RESULTS / "native_matchers" / f"{model}_{kind}_s{strength:03d}/matching_summary.json"
                    for model in ("loftr","roma") for kind,strength in (("clean",0),("texture",80))]
        deadline = time.monotonic()+12*3600
        while not all(path.exists() for path in expected):
            if time.monotonic()>deadline:
                raise TimeoutError("Native matcher queue did not complete")
            time.sleep(20)
    config = load_config(ROOT / "configs/default.toml")
    pairs = DATA / "manifests/hpatches/hpatches_pairs.csv"
    output = RESULTS / "learned_texture_nsi"
    output.mkdir(exist_ok=True)
    for identifier, arguments in MODELS:
        if identifier == "sift_default":
            continue
        base_args = build_parser().parse_args(["build-hpatches-model",*arguments,"--export-id",identifier,
            "--pairs-manifest",str(pairs),"--dataset-root",str(DATA / "datasets/HPatches/hpatches-sequences-release"),
            "--cache-root",str(DATA / "cache"),"--output",str(DATA / "cache/unused.npz"),"--device","cuda"])
        print(f"Learned texture START: {identifier}",flush=True)
        adapter = _batch_adapter(base_args)
        for strength in (.2,.4,.6,.8,1.):
            export_id = f"{identifier}_texturegrid_s{round(100*strength):03d}"
            result = output / (export_id+".json")
            if result.exists():
                continue
            features = DATA / "manifests/features" / (export_id+"_clean_attributes.npz")
            report = build_hpatches_feature_dataset(TextureAdapter(adapter,strength),pairs,
                base_args.dataset_root,features,DATA / "cache",export_id=export_id,max_samples_per_pair=64,
                seed=config.seed,expected_pairs=580,proxy="ncc",allow_empty_pairs=True)
            report.update({"scope":"exploratory texture grid after clean/ETH3D outcomes; not prospective confirmatory validation",
                "targets":"unchanged clean-image scalar NCC and Sobel attributes; probes refitted by strength on frozen81/18/17 split",
                "sampling":"conditional on surviving native features; pair/group coverage reported; no fixed-keypoint trajectory claim",
                "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            result.with_suffix(".build.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            payload = compute_nsi_artifacts(features,result,config)
            print(f"Learned texture DONE: {export_id}; NSI={payload['result']['NSI']:.6g}; aligned={report['pairs_with_aligned_features']}/580",flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
