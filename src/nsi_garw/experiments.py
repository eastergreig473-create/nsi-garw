from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd

from .config import KSGConfig, PipelineConfig, RidgeConfig, SubspaceConfig
from .garw import calibrate_garw, evaluate_garw, split_calibration_evaluation
from .adapters import validate_model_metadata
from .io import FeatureDataset, load_feature_npz, sha256_file
from .metrics import compute_nsi, groupwise_nsi_uncertainty
from .evaluation import evaluate_match_table
from .figures import generate_figures
from .runtime import measure_call
from .statistics import ranking_stability, spearman_group_association
from .local_proxy import fit_local_nsi_proxy


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _finite_json(payload), ensure_ascii=False, indent=2, sort_keys=True
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def config_sha256(config: PipelineConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "pandas", "matplotlib", "Pillow", "psutil", "h5py"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    try:
        versions["nsi-garw"] = importlib_metadata.version("nsi-garw")
    except importlib_metadata.PackageNotFoundError:
        versions["nsi-garw"] = "source-tree"
    return versions


def save_feature_npz(path: str | Path, dataset: FeatureDataset) -> None:
    dataset.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "features": dataset.features,
        "geometry": dataset.geometry,
        "appearance": dataset.appearance,
        "groups": dataset.groups.astype(str),
    }
    if dataset.split is not None:
        payload["split"] = dataset.split.astype(str)
    if dataset.sample_ids is not None:
        payload["sample_ids"] = dataset.sample_ids.astype(str)
    if dataset.metadata is not None:
        payload["metadata_json"] = np.asarray(
            json.dumps(dataset.metadata, ensure_ascii=False, sort_keys=True)
        )
    np.savez_compressed(destination, **payload)


def make_synthetic_feature_dataset(
    *,
    separated: bool,
    seed: int = 0,
    group_count: int = 15,
    samples_per_group: int = 32,
    feature_dimension: int = 24,
) -> FeatureDataset:
    """Create diagnostic data; never use this as manuscript evidence."""
    if group_count < 3 or samples_per_group < 4 or feature_dimension < 8:
        raise ValueError("Synthetic dimensions are too small")
    rng = np.random.default_rng(seed)
    n = group_count * samples_per_group
    groups = np.repeat([f"scene_{index:03d}" for index in range(group_count)], samples_per_group)
    geometry_latent = rng.normal(size=(n, 3))
    appearance_private = rng.normal(size=(n, 2))
    group_offsets = rng.normal(scale=0.15, size=(group_count, 3))
    geometry = geometry_latent + group_offsets[np.repeat(np.arange(group_count), samples_per_group)]
    geometry += rng.normal(scale=0.04, size=geometry.shape)
    features = rng.normal(scale=0.55, size=(n, feature_dimension))
    if separated:
        appearance = appearance_private + rng.normal(scale=0.04, size=(n, 2))
        features[:, :3] += 2.2 * geometry_latent
        features[:, 4:6] += 2.2 * appearance_private
        label = "separated"
    else:
        appearance = 0.82 * geometry_latent[:, :2] + 0.28 * appearance_private
        appearance += rng.normal(scale=0.04, size=(n, 2))
        shared = geometry_latent[:, :2] + 0.55 * appearance_private
        features[:, :2] += 2.2 * shared
        features[:, 2] += 2.0 * geometry_latent[:, 2]
        label = "entangled"
    sample_ids = np.asarray([f"{group}:{index:04d}" for index, group in enumerate(groups)])
    return FeatureDataset(
        features=features,
        geometry=geometry,
        appearance=appearance,
        groups=groups,
        sample_ids=sample_ids,
        metadata={
            "synthetic": True,
            "scientific_evidence": False,
            "diagnostic_condition": label,
            "seed": seed,
        },
    )


def make_synthetic_garw_table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for split, scene_start, scene_count in (("calibration", 0, 6), ("evaluation", 6, 8)):
        for scene_index in range(scene_start, scene_start + scene_count):
            scene = f"garw_scene_{scene_index:03d}"
            for query_index in range(10):
                query = f"{scene}_q{query_index:03d}"
                local_nsi = float(rng.uniform(0.05, 0.95))
                appearance_reliability = local_nsi
                geometry_reliability = 1.0 - local_nsi
                for candidate_index in range(8):
                    label = int(candidate_index == 0)
                    signal = 1.0 if label else -0.15
                    geometry = signal * (0.35 + 0.65 * geometry_reliability) + rng.normal(scale=0.24)
                    appearance = signal * (0.35 + 0.65 * appearance_reliability) + rng.normal(scale=0.24)
                    rows.append(
                        {
                            "query_id": query,
                            "scene": scene,
                            "label": label,
                            "s_geometry": geometry,
                            "s_appearance": appearance,
                            "nsi_local": local_nsi,
                            "nsi_mode": "non_oracle_proxy",
                            "split": split,
                        }
                    )
    return pd.DataFrame(rows)


def compute_nsi_artifacts(
    input_path: str | Path,
    output_json: str | Path,
    config: PipelineConfig,
    *,
    projector_output: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    execution_source_sha256 = source_sha256()
    execution_input_sha256 = sha256_file(source)
    dataset = load_feature_npz(source)
    def _compute_complete_nsi():
        result = compute_nsi(dataset, config)
        uncertainty = groupwise_nsi_uncertainty(dataset, result, config)
        return result, uncertainty

    (result, uncertainty), runtime = measure_call(_compute_complete_nsi)
    payload = {
        "result": result.to_dict(),
        "groupwise_uncertainty": uncertainty,
        "provenance": {
            "input_path": str(source.resolve()),
            "input_sha256": execution_input_sha256,
            "config": asdict(config),
            "config_sha256": config_sha256(config),
            "source_sha256": execution_source_sha256,
            "source_sha256_after_execution": source_sha256(),
            "software_versions": software_versions(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime.to_dict(),
            "reimplementation": True,
        },
    }
    write_json(output_json, payload)
    if projector_output is not None:
        projector_path = Path(projector_output)
        projector_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            projector_path,
            geometry_basis=result.geometry_basis,
            appearance_raw_basis=result.appearance_raw_basis,
            appearance_projected_basis=result.appearance_projected_basis,
            geometry_coefficients=result.geometry_probe.coefficients_standardized,
            appearance_coefficients=result.appearance_probe.coefficients_standardized,
            feature_mean=result.geometry_probe.x_mean,
            feature_scale=result.geometry_probe.x_scale,
            coordinate_system=np.asarray("training-standardized feature coordinates"),
        )
    return payload


def run_batch_manifest(
    manifest_path: str | Path,
    output_directory: str | Path,
    config: PipelineConfig,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {
        "experiment_id",
        "model",
        "dataset",
        "layer",
        "condition",
        "strength",
        "input_npz",
        "implementation",
        "implementation_commit",
        "checkpoint",
        "checkpoint_sha256",
        "preprocessing",
        "coordinate_system",
        "annotation_source",
        "split_id",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest["experiment_id"].astype(str).duplicated().any():
        raise ValueError("experiment_id values must be unique")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_root = Path(manifest_path).resolve().parent
    manifest_hash = sha256_file(manifest_path)
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        input_path = Path(str(record["input_npz"]))
        if not input_path.is_absolute():
            input_path = manifest_root / input_path
        source_dataset = load_feature_npz(input_path)
        validate_model_metadata(source_dataset.metadata or {})
        for key in ("model", "layer", "implementation_commit", "checkpoint_sha256"):
            if str(source_dataset.metadata.get(key, "")) != str(record[key]):
                raise ValueError(
                    f"Manifest/NPZ metadata mismatch for {record['experiment_id']!r}: {key}"
                )
        experiment_id = str(record["experiment_id"])
        result_path = output / f"{experiment_id}.json"
        projector_path = output / f"{experiment_id}_projectors.npz"
        payload = compute_nsi_artifacts(
            input_path,
            result_path,
            config,
            projector_output=projector_path,
        )
        result = payload["result"]
        rows.append(
            {
                **record,
                "A_accessibility": result["A_accessibility"],
                "B_separation": result["B_separation"],
                "C_information_purity": result["C_information_purity"],
                "NSI": result["NSI"],
                "OSR": result["OSR"],
                "geometry_r2": result["geometry_r2"],
                "appearance_r2": result["appearance_r2"],
                "result_json": str(result_path.resolve()),
                "projectors_npz": str(projector_path.resolve()),
                "manifest_sha256": manifest_hash,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "nsi_results.csv", index=False)
    write_json(
        output / "batch_provenance.json",
        {
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": manifest_hash,
            "config": asdict(config),
            "config_sha256": config_sha256(config),
            "source_sha256": source_sha256(),
            "software_versions": software_versions(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_count": int(len(summary)),
        },
    )
    return summary


def run_sensitivity(
    input_path: str | Path,
    output_csv: str | Path,
    config: PipelineConfig,
) -> pd.DataFrame:
    dataset = load_feature_npz(input_path)
    variants: list[tuple[str, str, PipelineConfig]] = [("baseline", "default", config)]
    for rank in sorted({1, 2, 4, config.subspace.max_rank}):
        variants.append(
            (
                "subspace_max_rank",
                str(rank),
                replace(config, subspace=replace(config.subspace, max_rank=rank)),
            )
        )
    for k in sorted({3, 5, 10, config.ksg.k}):
        variants.append(("ksg_k", str(k), replace(config, ksg=replace(config.ksg, k=k))))
    for transform in ("reciprocal_excess_nats", "exp_negative_excess_nats"):
        variants.append(
            (
                "purity_transform",
                transform,
                replace(config, ksg=replace(config.ksg, purity_transform=transform)),
            )
        )
    for overlap_metric in (
        "mean_squared_principal_cosine",
        "maximum_squared_principal_cosine",
    ):
        variants.append(
            (
                "osr_implementation",
                overlap_metric,
                replace(
                    config,
                    subspace=replace(config.subspace, overlap_metric=overlap_metric),
                ),
            )
        )
    for seed in sorted({config.seed, config.seed + 1, config.seed + 2}):
        variants.append(("seed", str(seed), replace(config, seed=seed)))
    for alpha in config.ridge.lambdas:
        variants.append(
            (
                "fixed_ridge_lambda",
                f"{alpha:g}",
                replace(config, ridge=RidgeConfig(lambdas=(float(alpha),))),
            )
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, value, variant in variants:
        identity = config_sha256(variant)
        if identity in seen:
            continue
        seen.add(identity)
        result = compute_nsi(dataset, variant)
        rows.append(
            {
                "family": family,
                "value": value,
                "variant": "default" if family == "baseline" else f"{family}={value}",
                "config_sha256": identity,
                "A_accessibility": result.accessibility,
                "B_separation": result.separation,
                "C_information_purity": result.information_purity,
                "NSI": result.nsi,
                "OSR": result.osr,
                "geometry_r2": result.geometry_r2,
                "appearance_r2": result.appearance_r2,
                "geometry_rank": result.geometry_rank,
                "appearance_raw_rank": result.appearance_raw_rank,
            }
        )
    table = pd.DataFrame(rows)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


def run_sensitivity_manifest(
    manifest_path: str | Path,
    output_directory: str | Path,
    config: PipelineConfig,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"experiment_id", "model", "dataset", "annotation_source", "input_npz"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Sensitivity manifest is missing columns: {missing}")
    if manifest["experiment_id"].astype(str).duplicated().any():
        raise ValueError("Sensitivity experiment_id values must be unique")
    root = Path(manifest_path).resolve().parent
    output = Path(output_directory)
    per_experiment = output / "by_experiment"
    per_experiment.mkdir(parents=True, exist_ok=True)
    blocks: list[pd.DataFrame] = []
    for record in manifest.to_dict(orient="records"):
        source = Path(str(record["input_npz"]))
        if not source.is_absolute():
            source = root / source
        destination = per_experiment / f"{record['experiment_id']}.csv"
        table = run_sensitivity(source, destination, config)
        for column in ("experiment_id", "model", "dataset", "annotation_source"):
            table.insert(0, column, record[column])
        table["input_npz"] = str(source.resolve())
        table["input_sha256"] = sha256_file(source)
        table.to_csv(destination, index=False)
        blocks.append(table)
    combined = pd.concat(blocks, ignore_index=True)
    combined.to_csv(output / "sensitivity_results.csv", index=False)
    write_json(
        output / "sensitivity_provenance.json",
        {
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "config_sha256": config_sha256(config),
            "source_sha256": source_sha256(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_count": int(len(manifest)),
            "result_rows": int(len(combined)),
        },
    )
    return combined


def run_garw_table(
    input_csv: str | Path,
    output_directory: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    table = pd.read_csv(input_csv)
    calibration, evaluation = split_calibration_evaluation(table)
    parameters, trace = calibrate_garw(
        calibration, config.garw.penalty_lambdas, seed=config.seed
    )
    lambda_zero_parameters, lambda_zero_trace = calibrate_garw(
        calibration, (0.0,), seed=config.seed
    )
    result = evaluate_garw(
        evaluation,
        parameters,
        bootstrap_replicates=config.statistics.bootstrap_replicates,
        confidence_level=config.statistics.confidence_level,
        seed=config.seed,
        lambda_zero_parameters=lambda_zero_parameters,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.per_query.to_csv(output / "garw_per_query.csv", index=False)
    result.per_scene.to_csv(output / "garw_per_scene.csv", index=False)
    payload = result.to_dict()
    payload["calibration_trace"] = trace
    payload["lambda_zero_calibration_trace"] = lambda_zero_trace
    payload["lambda_zero_parameters"] = lambda_zero_parameters.to_dict()
    payload["provenance"] = {
        "input_path": str(Path(input_csv).resolve()),
        "input_sha256": sha256_file(input_csv),
        "config_sha256": config_sha256(config),
        "source_sha256": source_sha256(),
        "software_versions": software_versions(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reimplementation": True,
        "conflict_penalty_status": "declared reconstruction choice",
        "nsi_mode": str(table["nsi_mode"].iloc[0]),
    }
    write_json(output / "garw_summary.json", payload)
    return payload


def run_local_proxy(
    input_csv: str | Path,
    output_csv: str | Path,
    model_json: str | Path,
    config: PipelineConfig,
    *,
    feature_columns: tuple[str, ...],
    target_column: str = "nsi_reference",
    group_column: str = "scene",
) -> dict[str, Any]:
    table = pd.read_csv(input_csv)
    output, model, trace = fit_local_nsi_proxy(
        table,
        feature_columns=feature_columns,
        target_column=target_column,
        group_column=group_column,
        lambdas=config.ridge.lambdas,
        seed=config.seed,
    )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    payload = {
        "model": model.to_dict(),
        "validation_trace": trace,
        "provenance": {
            "input_path": str(Path(input_csv).resolve()),
            "input_sha256": sha256_file(input_csv),
            "config_sha256": config_sha256(config),
            "source_sha256": source_sha256(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_role": "non-oracle observable proxy for local NSI",
        },
    }
    write_json(model_json, payload)
    return payload


def run_match_evaluation(
    input_csv: str | Path,
    output_directory: str | Path,
    config: PipelineConfig,
    *,
    thresholds: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0),
    primary_method: str | None = None,
    baseline_methods: tuple[str, ...] = (),
    symmetric_homography: bool = False,
) -> dict[str, Any]:
    table = pd.read_csv(input_csv)
    result, runtime = measure_call(
        evaluate_match_table,
        table,
        thresholds=thresholds,
        primary_method=primary_method,
        baseline_methods=baseline_methods,
        symmetric_homography=symmetric_homography,
        bootstrap_replicates=config.statistics.bootstrap_replicates,
        confidence_level=config.statistics.confidence_level,
        seed=config.seed,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.per_match.to_csv(output / "matching_per_match.csv", index=False)
    result.per_pair.to_csv(output / "matching_per_pair.csv", index=False)
    result.per_scene.to_csv(output / "matching_per_scene.csv", index=False)
    payload = result.to_dict()
    payload["provenance"] = {
        "input_path": str(Path(input_csv).resolve()),
        "input_sha256": sha256_file(input_csv),
        "config_sha256": config_sha256(config),
        "source_sha256": source_sha256(),
        "thresholds": list(thresholds),
        "symmetric_homography": symmetric_homography,
        "runtime": runtime.to_dict(),
        "software_versions": software_versions(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reimplementation": True,
    }
    write_json(output / "matching_summary.json", payload)
    return payload


def run_association_analysis(
    input_csv: str | Path,
    output_json: str | Path,
    config: PipelineConfig,
    *,
    x_column: str,
    y_column: str,
    group_column: str | None,
) -> dict[str, Any]:
    table = pd.read_csv(input_csv)
    required = {x_column, y_column}
    if group_column is not None:
        required.add(group_column)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Association table is missing columns: {missing}")
    groups = table[group_column].to_numpy() if group_column is not None else None
    result = spearman_group_association(
        table[x_column].to_numpy(dtype=np.float64),
        table[y_column].to_numpy(dtype=np.float64),
        groups,
        bootstrap_replicates=config.statistics.bootstrap_replicates,
        permutation_replicates=config.statistics.permutation_replicates,
        confidence_level=config.statistics.confidence_level,
        seed=config.seed,
    )
    payload = {
        "analysis": result.to_dict(),
        "x_column": x_column,
        "y_column": y_column,
        "group_column": group_column,
        "provenance": {
            "input_path": str(Path(input_csv).resolve()),
            "input_sha256": sha256_file(input_csv),
            "config_sha256": config_sha256(config),
            "source_sha256": source_sha256(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(output_json, payload)
    return payload


def run_rank_stability_analysis(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    item_column: str,
    variant_column: str,
    score_column: str,
    baseline_variant: str,
) -> pd.DataFrame:
    table = pd.read_csv(input_csv)
    required = {item_column, variant_column, score_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Rank-stability table is missing columns: {missing}")
    if table.duplicated([item_column, variant_column]).any():
        raise ValueError("Each item/variant pair must be unique")
    pivot = table.pivot(index=item_column, columns=variant_column, values=score_column)
    if baseline_variant not in pivot.columns:
        raise ValueError(f"Baseline variant {baseline_variant!r} is absent")
    complete = pivot.dropna(axis=0, how="any")
    if len(complete) < 3:
        raise ValueError("At least three items with all variants are required")
    variants = {
        str(column): complete[column].to_numpy(dtype=np.float64)
        for column in complete.columns
        if str(column) != baseline_variant
    }
    results = ranking_stability(
        complete[baseline_variant].to_numpy(dtype=np.float64), variants
    )
    output = pd.DataFrame([result.to_dict() for result in results])
    output.insert(0, "baseline_variant", baseline_variant)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return output


def run_smoke(output_directory: str | Path, config: PipelineConfig) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    smoke_config = replace(
        config,
        ksg=KSGConfig(
            k=min(config.ksg.k, 5),
            permutations=min(config.ksg.permutations, 12),
            max_samples=min(config.ksg.max_samples, 480),
            jitter=config.ksg.jitter,
            purity_transform=config.ksg.purity_transform,
        ),
        statistics=replace(
            config.statistics,
            bootstrap_replicates=min(config.statistics.bootstrap_replicates, 300),
            permutation_replicates=min(config.statistics.permutation_replicates, 100),
        ),
    )
    nsi_payload: dict[str, Any] = {}
    nsi_rows: list[dict[str, Any]] = []
    for offset, condition in enumerate(("separated", "entangled")):
        dataset = make_synthetic_feature_dataset(
            separated=condition == "separated", seed=config.seed + offset
        )
        data_path = output / f"synthetic_{condition}.npz"
        result_path = output / f"synthetic_{condition}.json"
        save_feature_npz(data_path, dataset)
        nsi_payload[condition] = compute_nsi_artifacts(
            data_path,
            result_path,
            smoke_config,
            projector_output=output / f"synthetic_{condition}_projectors.npz",
        )["result"]
        nsi_rows.append(
            {
                "experiment_id": f"synthetic_{condition}",
                "model": f"synthetic_{condition}",
                "dataset": "synthetic_smoke_only",
                "layer": "diagnostic",
                "condition": condition,
                "strength": 0.0,
                "A_accessibility": nsi_payload[condition]["A_accessibility"],
                "B_separation": nsi_payload[condition]["B_separation"],
                "C_information_purity": nsi_payload[condition]["C_information_purity"],
                "NSI": nsi_payload[condition]["NSI"],
                "OSR": nsi_payload[condition]["OSR"],
            }
        )
    pd.DataFrame(nsi_rows).to_csv(output / "nsi_results.csv", index=False)
    garw_table = make_synthetic_garw_table(seed=config.seed)
    garw_path = output / "synthetic_garw_candidates.csv"
    garw_table.to_csv(garw_path, index=False)
    garw_payload = run_garw_table(garw_path, output / "garw", smoke_config)
    figure_payload = generate_figures(output, output / "figures")
    checks = {
        "separated_nsi_exceeds_entangled": bool(
            nsi_payload["separated"]["NSI"] > nsi_payload["entangled"]["NSI"]
        ),
        "osr_bounded": bool(
            all(0.0 <= nsi_payload[name]["OSR"] <= 1.0 for name in nsi_payload)
        ),
        "garw_output_present": "summary" in garw_payload,
        "figure_output_present": bool(figure_payload.get("figures")),
    }
    payload = {
        "scientific_evidence": False,
        "purpose": "implementation smoke test only",
        "checks": checks,
        "nsi": nsi_payload,
        "garw": garw_payload,
        "figures": figure_payload,
        "passed": bool(all(checks.values())),
    }
    write_json(output / "smoke_report.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"Smoke test failed: {checks}")
    return payload
