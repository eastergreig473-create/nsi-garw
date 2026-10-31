from __future__ import annotations

import argparse
import json

from nsi_garw.cli import _batch_adapter, build_parser
from nsi_garw.matching_builders import build_hpatches_descriptor_matching
from run_hpatches_suite import DATA, MODELS, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=["sift_default"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for identifier, adapter_args in MODELS:
        if identifier not in args.only:
            continue
        output = RESULTS / "descriptor_matching" / identifier
        arguments = build_parser().parse_args(["build-hpatches-model", *adapter_args,
            "--export-id", identifier, "--pairs-manifest", str(DATA / "manifests/hpatches/hpatches_pairs.csv"),
            "--dataset-root", str(DATA / "datasets/HPatches/hpatches-sequences-release"),
            "--cache-root", str(DATA / "cache"), "--output", str(DATA / "cache/unused.npz"),
            "--device", args.device])
        adapter = _batch_adapter(arguments)
        payload = build_hpatches_descriptor_matching(adapter, arguments.pairs_manifest,
            arguments.dataset_root, arguments.cache_root, output, export_id=identifier)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
