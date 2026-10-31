from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

from .config import load_config
from .degradations import DEGRADATIONS, apply_degradation
from .experiments import (
    compute_nsi_artifacts,
    run_batch_manifest,
    run_garw_table,
    run_match_evaluation,
    run_association_analysis,
    run_rank_stability_analysis,
    run_local_proxy,
    run_sensitivity,
    run_sensitivity_manifest,
    run_smoke,
)
from .figures import generate_figures
from .io import load_feature_npz, sha256_file
from .annotation_io import save_depth_attributes, save_hpatches_attributes
from .adapters import (
    DeDoDeAdapter,
    DINOv2Adapter,
    LocalFeatures,
    LoFTRBackboneAdapter,
    RoMaEncoderAdapter,
    VGGTAggregatorAdapter,
    CroCoEncoderAdapter,
    ResNet50Adapter,
    SIFTAdapter,
    SuperPointAdapter,
    XFeatAdapter,
)
from .full_rerun import preflight_full_rerun, run_full_rerun
from .datasets import build_hpatches_manifests
from .feature_builders import build_hpatches_feature_dataset


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML configuration (defaults to the built-in declared protocol)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsi-garw",
        description="Auditable clean-room reimplementation of the NSI/GARW experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-npz", help="Validate a canonical feature archive")
    validate.add_argument("--input", type=Path, required=True)

    compute = subparsers.add_parser("compute-nsi", help="Compute NSI for one feature archive")
    compute.add_argument("--input", type=Path, required=True)
    compute.add_argument("--output", type=Path, required=True)
    compute.add_argument("--projectors", type=Path, default=None)
    _config_argument(compute)

    batch = subparsers.add_parser("batch-nsi", help="Run all rows in an experiment manifest")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    _config_argument(batch)

    sensitivity = subparsers.add_parser("sensitivity", help="Run declared NSI sensitivity analyses")
    sensitivity.add_argument("--input", type=Path, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)
    _config_argument(sensitivity)

    sensitivity_batch = subparsers.add_parser("sensitivity-batch", help="Run sensitivity grids across models, datasets, and annotation sources")
    sensitivity_batch.add_argument("--manifest", type=Path, required=True)
    sensitivity_batch.add_argument("--output-dir", type=Path, required=True)
    _config_argument(sensitivity_batch)

    garw = subparsers.add_parser("garw", help="Calibrate and evaluate GARW with scene isolation")
    garw.add_argument("--input", type=Path, required=True)
    garw.add_argument("--output-dir", type=Path, required=True)
    _config_argument(garw)

    matches = subparsers.add_parser("evaluate-matches", help="Compute homography MMA and paired scene statistics")
    matches.add_argument("--input", type=Path, required=True)
    matches.add_argument("--output-dir", type=Path, required=True)
    matches.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0])
    matches.add_argument("--primary-method", default=None)
    matches.add_argument("--baseline-method", action="append", default=[])
    matches.add_argument("--symmetric-homography", action="store_true")
    _config_argument(matches)

    association = subparsers.add_parser("association", help="Spearman association with group bootstrap and permutation")
    association.add_argument("--input", type=Path, required=True)
    association.add_argument("--output", type=Path, required=True)
    association.add_argument("--x-column", required=True)
    association.add_argument("--y-column", required=True)
    association.add_argument("--group-column", default=None)
    _config_argument(association)

    stability = subparsers.add_parser("rank-stability", help="Compare model/layer rankings across protocol variants")
    stability.add_argument("--input", type=Path, required=True)
    stability.add_argument("--output", type=Path, required=True)
    stability.add_argument("--item-column", default="model")
    stability.add_argument("--variant-column", default="variant")
    stability.add_argument("--score-column", default="NSI")
    stability.add_argument("--baseline-variant", default="default")
    _config_argument(stability)

    proxy = subparsers.add_parser("fit-local-proxy", help="Fit a scene-CV observable proxy for local NSI")
    proxy.add_argument("--input", type=Path, required=True)
    proxy.add_argument("--output", type=Path, required=True)
    proxy.add_argument("--model-output", type=Path, required=True)
    proxy.add_argument("--feature-column", action="append", required=True)
    proxy.add_argument("--target-column", default="nsi_reference")
    proxy.add_argument("--group-column", default="scene")
    _config_argument(proxy)

    figures = subparsers.add_parser("figures", help="Generate publication-format result figures")
    figures.add_argument("--results-dir", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="Run non-scientific synthetic implementation checks")
    smoke.add_argument("--output-dir", type=Path, required=True)
    _config_argument(smoke)

    full = subparsers.add_parser(
        "full-rerun",
        help="Preflight or execute the complete frozen scientific experiment suite",
    )
    full.add_argument("--plan", type=Path, required=True)
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument(
        "--execute",
        action="store_true",
        help="Execute after a successful asset/manifest preflight; otherwise preflight only",
    )
    _config_argument(full)

    degrade = subparsers.add_parser("degrade-image", help="Apply one deterministic stress condition")
    degrade.add_argument("--input", type=Path, required=True)
    degrade.add_argument("--output", type=Path, required=True)
    degrade.add_argument("--kind", choices=sorted(DEGRADATIONS), required=True)
    degrade.add_argument("--strength", type=float, required=True)
    degrade.add_argument("--seed", type=int, default=20261023)

    depth_attributes = subparsers.add_parser("attributes-depth", help="Build normal and Sobel maps from an image/depth pair")
    depth_attributes.add_argument("--image", type=Path, required=True)
    depth_attributes.add_argument("--depth", type=Path, required=True)
    depth_attributes.add_argument("--output", type=Path, required=True)
    depth_attributes.add_argument("--depth-key", default=None)
    depth_attributes.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), default=None)

    hpatches_attributes = subparsers.add_parser("attributes-hpatches", help="Build homography-consistency and Sobel maps")
    hpatches_attributes.add_argument("--reference", type=Path, required=True)
    hpatches_attributes.add_argument("--target", type=Path, required=True)
    hpatches_attributes.add_argument("--homography", type=Path, required=True)
    hpatches_attributes.add_argument("--output", type=Path, required=True)
    hpatches_attributes.add_argument("--proxy", choices=["ncc", "cosine"], default="ncc")

    resnet_features = subparsers.add_parser(
        "extract-resnet50",
        help="Export a pinned torchvision ResNet-50 feature map without downloading weights",
    )
    resnet_features.add_argument("--image", type=Path, required=True)
    resnet_features.add_argument("--checkpoint", type=Path, required=True)
    resnet_features.add_argument(
        "--layer",
        choices=["maxpool", "layer1", "layer2", "layer3", "layer4"],
        required=True,
    )
    resnet_features.add_argument("--device", default=None)
    resnet_features.add_argument("--output", type=Path, required=True)

    superpoint_features = subparsers.add_parser(
        "extract-superpoint",
        help="Export pinned LightGlue-repository SuperPoint descriptors without network access",
    )
    superpoint_features.add_argument("--image", type=Path, required=True)
    superpoint_features.add_argument("--repository", type=Path, required=True)
    superpoint_features.add_argument("--checkpoint", type=Path, required=True)
    superpoint_features.add_argument("--max-keypoints", type=int, default=4096)
    superpoint_features.add_argument("--resize", type=int, default=1024)
    superpoint_features.add_argument("--device", default=None)
    superpoint_features.add_argument("--output", type=Path, required=True)

    xfeat_features = subparsers.add_parser(
        "extract-xfeat",
        help="Export pinned repository XFeat sparse descriptors without network access",
    )
    xfeat_features.add_argument("--image", type=Path, required=True)
    xfeat_features.add_argument("--repository", type=Path, required=True)
    xfeat_features.add_argument("--checkpoint", type=Path, required=True)
    xfeat_features.add_argument("--max-keypoints", type=int, default=4096)
    xfeat_features.add_argument("--detection-threshold", type=float, default=0.05)
    xfeat_features.add_argument("--device", default=None)
    xfeat_features.add_argument("--output", type=Path, required=True)

    dinov2_features = subparsers.add_parser(
        "extract-dinov2",
        help="Export pinned DINOv2 ViT-S/14 patch tokens without network access",
    )
    dinov2_features.add_argument("--image", type=Path, required=True)
    dinov2_features.add_argument("--repository", type=Path, required=True)
    dinov2_features.add_argument("--checkpoint", type=Path, required=True)
    dinov2_features.add_argument("--layer", type=int, choices=range(12), required=True)
    dinov2_features.add_argument("--max-size", type=int, default=518)
    dinov2_features.add_argument("--device", default=None)
    dinov2_features.add_argument("--output", type=Path, required=True)

    dedode_features = subparsers.add_parser(
        "extract-dedode",
        help="Export pinned DeDoDe-L-v2/descriptor-B features without network access",
    )
    dedode_features.add_argument("--image", type=Path, required=True)
    dedode_features.add_argument("--repository", type=Path, required=True)
    dedode_features.add_argument("--detector-checkpoint", type=Path, required=True)
    dedode_features.add_argument("--descriptor-checkpoint", type=Path, required=True)
    dedode_features.add_argument("--max-keypoints", type=int, default=4096)
    dedode_features.add_argument("--resize", type=int, default=784)
    dedode_features.add_argument("--device", default=None)
    dedode_features.add_argument("--output", type=Path, required=True)

    loftr_features = subparsers.add_parser(
        "extract-loftr",
        help="Export pinned LoFTR backbone tensors before matching",
    )
    loftr_features.add_argument("--image", type=Path, required=True)
    loftr_features.add_argument("--checkpoint", type=Path, required=True)
    loftr_features.add_argument(
        "--representation", choices=["coarse", "fine"], default="coarse"
    )
    loftr_features.add_argument("--max-size", type=int, default=1024)
    loftr_features.add_argument("--device", default=None)
    loftr_features.add_argument("--output", type=Path, required=True)

    roma_features = subparsers.add_parser(
        "extract-roma",
        help="Export a pinned RoMa encoder feature-pyramid tensor without network access",
    )
    roma_features.add_argument("--image", type=Path, required=True)
    roma_features.add_argument("--repository", type=Path, required=True)
    roma_features.add_argument("--checkpoint", type=Path, required=True)
    roma_features.add_argument("--dinov2-checkpoint", type=Path, required=True)
    roma_features.add_argument("--scale", type=int, choices=[1, 2, 4, 8, 16], default=16)
    roma_features.add_argument("--resolution", type=int, default=560)
    roma_features.add_argument("--device", default=None)
    roma_features.add_argument("--output", type=Path, required=True)

    vggt_features = subparsers.add_parser(
        "extract-vggt",
        help="Export pinned VGGT aggregator patch tokens without network access",
    )
    vggt_features.add_argument("--image", type=Path, required=True)
    vggt_features.add_argument("--repository", type=Path, required=True)
    vggt_features.add_argument("--checkpoint", type=Path, required=True)
    vggt_features.add_argument("--layer", type=int, choices=[4, 11, 17, 23], default=23)
    vggt_features.add_argument("--image-size", type=int, default=518)
    vggt_features.add_argument("--device", default=None)
    vggt_features.add_argument("--output", type=Path, required=True)

    for command_name, model_type in (
        ("extract-mast3r", "mast3r"),
        ("extract-dust3r", "dust3r"),
    ):
        croco_features = subparsers.add_parser(
            command_name,
            help=f"Export pinned {model_type.upper()} single-image encoder tokens",
        )
        croco_features.set_defaults(croco_model_type=model_type)
        croco_features.add_argument("--image", type=Path, required=True)
        croco_features.add_argument("--repository", type=Path, required=True)
        croco_features.add_argument("--checkpoint", type=Path, required=True)
        croco_features.add_argument("--layer", type=int, default=23)
        croco_features.add_argument("--image-size", type=int, default=512)
        croco_features.add_argument("--device", default=None)
        croco_features.add_argument("--output", type=Path, required=True)

    prepare_hpatches = subparsers.add_parser(
        "prepare-hpatches",
        help="Validate HPatches and emit deterministic sequence-isolated manifests",
    )
    prepare_hpatches.add_argument("--dataset-root", type=Path, required=True)
    prepare_hpatches.add_argument("--output-dir", type=Path, required=True)
    prepare_hpatches.add_argument("--seed", type=int, default=20261023)
    prepare_hpatches.add_argument("--train-fraction", type=float, default=0.70)
    prepare_hpatches.add_argument("--validation-fraction", type=float, default=0.15)

    hpatches_model = subparsers.add_parser(
        "build-hpatches-model",
        help="Extract one model across HPatches and build an aligned NSI dataset",
    )
    hpatches_model.add_argument(
        "--adapter",
        choices=[
            "sift",
            "superpoint",
            "xfeat",
            "dedode",
            "loftr",
            "roma",
            "dinov2",
            "vggt",
            "mast3r",
            "dust3r",
            "resnet50",
        ],
        required=True,
    )
    hpatches_model.add_argument("--export-id", required=True)
    hpatches_model.add_argument("--pairs-manifest", type=Path, required=True)
    hpatches_model.add_argument("--dataset-root", type=Path, required=True)
    hpatches_model.add_argument("--cache-root", type=Path, required=True)
    hpatches_model.add_argument("--output", type=Path, required=True)
    hpatches_model.add_argument("--repository", type=Path, default=None)
    hpatches_model.add_argument("--checkpoint", type=Path, default=None)
    hpatches_model.add_argument("--aux-checkpoint", type=Path, default=None)
    hpatches_model.add_argument("--layer", default=None)
    hpatches_model.add_argument("--image-size", type=int, default=None)
    hpatches_model.add_argument("--max-keypoints", type=int, default=4096)
    hpatches_model.add_argument("--threshold", type=float, default=0.05)
    hpatches_model.add_argument("--device", default=None)
    hpatches_model.add_argument("--max-samples-per-pair", type=int, default=64)
    hpatches_model.add_argument("--seed", type=int, default=20261023)
    hpatches_model.add_argument("--expected-pairs", type=int, default=580)
    hpatches_model.add_argument("--proxy", choices=["ncc", "cosine"], default="ncc")
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _save_local_features(features: LocalFeatures, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coordinates_xy=features.coordinates_xy,
        descriptors=features.descriptors,
        scores=features.scores,
        metadata_json=np.asarray(json.dumps(features.metadata, ensure_ascii=False)),
    )
    _print_json(
        {
            "output": str(output.resolve()),
            "samples": int(features.descriptors.shape[0]),
            "feature_dimension": int(features.descriptors.shape[1]),
            "metadata": features.metadata,
        }
    )


def _required_path(arguments: argparse.Namespace, name: str) -> Path:
    value = getattr(arguments, name)
    if value is None:
        option = "--" + name.replace("_", "-")
        raise ValueError(f"{option} is required for adapter {arguments.adapter}")
    return value


def _batch_adapter(arguments: argparse.Namespace):
    adapter = arguments.adapter
    device = arguments.device
    if adapter == "sift":
        return SIFTAdapter(max_keypoints=arguments.max_keypoints)
    if adapter == "superpoint":
        return SuperPointAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            max_keypoints=arguments.max_keypoints,
            resize=arguments.image_size or 1024,
            device=device,
        )
    if adapter == "xfeat":
        return XFeatAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            max_keypoints=arguments.max_keypoints,
            detection_threshold=arguments.threshold,
            device=device,
        )
    if adapter == "dedode":
        return DeDoDeAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            _required_path(arguments, "aux_checkpoint"),
            max_keypoints=arguments.max_keypoints,
            resize=arguments.image_size or 784,
            device=device,
        )
    if adapter == "loftr":
        return LoFTRBackboneAdapter(
            _required_path(arguments, "checkpoint"),
            representation=arguments.layer or "coarse",
            max_size=arguments.image_size or 1024,
            device=device,
        )
    if adapter == "roma":
        return RoMaEncoderAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            _required_path(arguments, "aux_checkpoint"),
            scale=int(arguments.layer or 16),
            resolution=arguments.image_size or 560,
            device=device,
        )
    if adapter == "dinov2":
        return DINOv2Adapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            layer=int(arguments.layer or 11),
            max_size=arguments.image_size or 518,
            device=device,
        )
    if adapter == "vggt":
        return VGGTAggregatorAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            layer=int(arguments.layer or 23),
            image_size=arguments.image_size or 518,
            device=device,
        )
    if adapter in {"mast3r", "dust3r"}:
        return CroCoEncoderAdapter(
            _required_path(arguments, "repository"),
            _required_path(arguments, "checkpoint"),
            model_type=adapter,
            layer=int(arguments.layer or 23),
            image_size=arguments.image_size or 512,
            device=device,
        )
    if adapter == "resnet50":
        return ResNet50Adapter(
            _required_path(arguments, "checkpoint"),
            layer=arguments.layer or "layer4",
            device=device,
        )
    raise ValueError(f"Unsupported adapter: {adapter}")


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate-npz":
        dataset = load_feature_npz(arguments.input)
        _print_json(
            {
                "valid": True,
                "input": str(arguments.input.resolve()),
                "sha256": sha256_file(arguments.input),
                "samples": int(dataset.features.shape[0]),
                "feature_dimension": int(dataset.features.shape[1]),
                "geometry_dimension": int(dataset.geometry.shape[1]),
                "appearance_dimension": int(dataset.appearance.shape[1]),
                "groups": int(np.unique(dataset.groups.astype(str)).size),
                "has_explicit_split": dataset.split is not None,
                "metadata": dataset.metadata,
            }
        )
        return 0
    if arguments.command == "figures":
        _print_json(generate_figures(arguments.results_dir, arguments.output_dir))
        return 0
    if arguments.command == "degrade-image":
        image = np.asarray(Image.open(arguments.input))
        degraded = apply_degradation(
            image, arguments.kind, arguments.strength, seed=arguments.seed
        )
        encoded = np.clip(np.rint(degraded * 255.0), 0, 255).astype(np.uint8)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded).save(arguments.output)
        _print_json(
            {
                "output": str(arguments.output.resolve()),
                "kind": arguments.kind,
                "strength": arguments.strength,
                "seed": arguments.seed,
            }
        )
        return 0
    if arguments.command == "attributes-depth":
        metadata = save_depth_attributes(
            arguments.image,
            arguments.depth,
            arguments.output,
            intrinsics=tuple(arguments.intrinsics) if arguments.intrinsics is not None else None,
            depth_npz_key=arguments.depth_key,
        )
        _print_json({"output": str(arguments.output.resolve()), "metadata": metadata})
        return 0
    if arguments.command == "attributes-hpatches":
        metadata = save_hpatches_attributes(
            arguments.reference,
            arguments.target,
            arguments.homography,
            arguments.output,
            proxy=arguments.proxy,
        )
        _print_json({"output": str(arguments.output.resolve()), "metadata": metadata})
        return 0
    if arguments.command == "extract-resnet50":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = ResNet50Adapter(
            arguments.checkpoint,
            layer=arguments.layer,
            device=arguments.device,
        )
        features = adapter.extract(image)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arguments.output,
            coordinates_xy=features.coordinates_xy,
            descriptors=features.descriptors,
            scores=features.scores,
            metadata_json=np.asarray(json.dumps(features.metadata, ensure_ascii=False)),
        )
        _print_json(
            {
                "output": str(arguments.output.resolve()),
                "samples": int(features.descriptors.shape[0]),
                "feature_dimension": int(features.descriptors.shape[1]),
                "metadata": features.metadata,
            }
        )
        return 0
    if arguments.command == "extract-superpoint":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = SuperPointAdapter(
            arguments.repository,
            arguments.checkpoint,
            max_keypoints=arguments.max_keypoints,
            resize=arguments.resize,
            device=arguments.device,
        )
        features = adapter.extract(image)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arguments.output,
            coordinates_xy=features.coordinates_xy,
            descriptors=features.descriptors,
            scores=features.scores,
            metadata_json=np.asarray(json.dumps(features.metadata, ensure_ascii=False)),
        )
        _print_json(
            {
                "output": str(arguments.output.resolve()),
                "samples": int(features.descriptors.shape[0]),
                "feature_dimension": int(features.descriptors.shape[1]),
                "metadata": features.metadata,
            }
        )
        return 0
    if arguments.command == "extract-xfeat":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = XFeatAdapter(
            arguments.repository,
            arguments.checkpoint,
            max_keypoints=arguments.max_keypoints,
            detection_threshold=arguments.detection_threshold,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "extract-dinov2":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = DINOv2Adapter(
            arguments.repository,
            arguments.checkpoint,
            layer=arguments.layer,
            max_size=arguments.max_size,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "extract-dedode":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = DeDoDeAdapter(
            arguments.repository,
            arguments.detector_checkpoint,
            arguments.descriptor_checkpoint,
            max_keypoints=arguments.max_keypoints,
            resize=arguments.resize,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "extract-loftr":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = LoFTRBackboneAdapter(
            arguments.checkpoint,
            representation=arguments.representation,
            max_size=arguments.max_size,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "extract-roma":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = RoMaEncoderAdapter(
            arguments.repository,
            arguments.checkpoint,
            arguments.dinov2_checkpoint,
            scale=arguments.scale,
            resolution=arguments.resolution,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "extract-vggt":
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = VGGTAggregatorAdapter(
            arguments.repository,
            arguments.checkpoint,
            layer=arguments.layer,
            image_size=arguments.image_size,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command in {"extract-mast3r", "extract-dust3r"}:
        image = np.asarray(Image.open(arguments.image).convert("RGB"))
        adapter = CroCoEncoderAdapter(
            arguments.repository,
            arguments.checkpoint,
            model_type=arguments.croco_model_type,
            layer=arguments.layer,
            image_size=arguments.image_size,
            device=arguments.device,
        )
        _save_local_features(adapter.extract(image), arguments.output)
        return 0
    if arguments.command == "prepare-hpatches":
        _print_json(
            build_hpatches_manifests(
                arguments.dataset_root,
                arguments.output_dir,
                seed=arguments.seed,
                train_fraction=arguments.train_fraction,
                validation_fraction=arguments.validation_fraction,
            )
        )
        return 0
    if arguments.command == "build-hpatches-model":
        adapter = _batch_adapter(arguments)
        _print_json(
            build_hpatches_feature_dataset(
                adapter,
                arguments.pairs_manifest,
                arguments.dataset_root,
                arguments.output,
                arguments.cache_root,
                export_id=arguments.export_id,
                max_samples_per_pair=arguments.max_samples_per_pair,
                seed=arguments.seed,
                expected_pairs=(
                    arguments.expected_pairs if arguments.expected_pairs > 0 else None
                ),
                proxy=arguments.proxy,
            )
        )
        return 0

    config = load_config(arguments.config)
    if arguments.command == "compute-nsi":
        payload = compute_nsi_artifacts(
            arguments.input,
            arguments.output,
            config,
            projector_output=arguments.projectors,
        )
        _print_json(payload)
    elif arguments.command == "batch-nsi":
        table = run_batch_manifest(arguments.manifest, arguments.output_dir, config)
        _print_json({"rows": len(table), "output": str(arguments.output_dir.resolve())})
    elif arguments.command == "sensitivity":
        table = run_sensitivity(arguments.input, arguments.output, config)
        _print_json({"rows": len(table), "output": str(arguments.output.resolve())})
    elif arguments.command == "sensitivity-batch":
        table = run_sensitivity_manifest(arguments.manifest, arguments.output_dir, config)
        _print_json({"rows": len(table), "output": str(arguments.output_dir.resolve())})
    elif arguments.command == "garw":
        _print_json(run_garw_table(arguments.input, arguments.output_dir, config))
    elif arguments.command == "evaluate-matches":
        _print_json(
            run_match_evaluation(
                arguments.input,
                arguments.output_dir,
                config,
                thresholds=tuple(arguments.thresholds),
                primary_method=arguments.primary_method,
                baseline_methods=tuple(arguments.baseline_method),
                symmetric_homography=arguments.symmetric_homography,
            )
        )
    elif arguments.command == "association":
        _print_json(
            run_association_analysis(
                arguments.input,
                arguments.output,
                config,
                x_column=arguments.x_column,
                y_column=arguments.y_column,
                group_column=arguments.group_column,
            )
        )
    elif arguments.command == "rank-stability":
        table = run_rank_stability_analysis(
            arguments.input,
            arguments.output,
            item_column=arguments.item_column,
            variant_column=arguments.variant_column,
            score_column=arguments.score_column,
            baseline_variant=arguments.baseline_variant,
        )
        _print_json({"rows": len(table), "output": str(arguments.output.resolve())})
    elif arguments.command == "fit-local-proxy":
        _print_json(
            run_local_proxy(
                arguments.input,
                arguments.output,
                arguments.model_output,
                config,
                feature_columns=tuple(arguments.feature_column),
                target_column=arguments.target_column,
                group_column=arguments.group_column,
            )
        )
    elif arguments.command == "smoke":
        payload = run_smoke(arguments.output_dir, config)
        _print_json({"passed": payload["passed"], "output": str(arguments.output_dir.resolve())})
    elif arguments.command == "full-rerun":
        if arguments.execute:
            _print_json(run_full_rerun(arguments.plan, arguments.output_dir, config))
        else:
            _print_json(preflight_full_rerun(arguments.plan))
    else:
        raise AssertionError(f"Unhandled command: {arguments.command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
