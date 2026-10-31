from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 model-export environment
    import tomli as tomllib  # type: ignore[no-redef]
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .experiments import (
    run_association_analysis,
    run_batch_manifest,
    run_garw_table,
    run_match_evaluation,
    run_rank_stability_analysis,
    run_sensitivity_manifest,
    config_sha256,
    software_versions,
    source_sha256,
    write_json,
)
from .figures import generate_figures
from .io import sha256_file


ASSET_COLUMNS = {
    "asset_id",
    "asset_type",
    "local_path",
    "source_url",
    "license_reviewed",
    "license_note",
    "repository_commit_or_release",
    "checkpoint_or_archive",
    "sha256",
    "acquired_utc",
    "preprocessing_or_coordinate_convention",
    "verified_by",
}

INPUT_KEYS = (
    "asset_registry",
    "nsi_manifest",
    "sensitivity_manifest",
    "garw_candidates",
    "matches",
    "association",
    "rank_variants",
)


def _read_plan(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("rb") as handle:
        plan = tomllib.load(handle)
    root = path.resolve().parent.parent
    return plan, root


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _placeholder(value: object) -> bool:
    text = str(value).strip().upper()
    return not text or "REPLACE" in text or text in {"TBD", "TODO", "PENDING"}


def validate_asset_registry(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        table = pd.read_csv(path, keep_default_na=False)
    except Exception as error:
        return [f"asset registry cannot be read: {error}"]
    missing = sorted(ASSET_COLUMNS.difference(table.columns))
    if missing:
        return [f"asset registry is missing columns: {missing}"]
    if table.empty:
        return ["asset registry contains no assets"]
    for index, record in enumerate(table.to_dict(orient="records"), start=2):
        asset_id = str(record.get("asset_id", f"row-{index}"))
        blank = sorted(column for column in ASSET_COLUMNS if _placeholder(record[column]))
        if blank:
            problems.append(f"asset {asset_id!r} has placeholder/blank fields: {blank}")
            continue
        if str(record["license_reviewed"]).strip().lower() not in {"yes", "true", "1"}:
            problems.append(f"asset {asset_id!r} has not passed license review")
        local_path = Path(str(record["local_path"]))
        if not local_path.is_absolute():
            problems.append(f"asset {asset_id!r} local_path is not absolute")
            continue
        if not local_path.exists():
            problems.append(f"asset {asset_id!r} is missing: {local_path}")
            continue
        if not local_path.is_file():
            problems.append(
                f"asset {asset_id!r} must point to the registered archive/checkpoint file, not a directory"
            )
            continue
        expected = str(record["sha256"]).strip().lower()
        actual = sha256_file(local_path).lower()
        if expected != actual:
            problems.append(
                f"asset {asset_id!r} SHA-256 mismatch: registry={expected}, actual={actual}"
            )
    return problems


def preflight_full_rerun(plan_path: str | Path) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan, root = _read_plan(path)
    problems: list[str] = []
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        problems.append("plan requires an [inputs] table")
        inputs = {}
    resolved: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key in INPUT_KEYS:
        value = inputs.get(key)
        if value is None or _placeholder(value):
            problems.append(f"plan input {key!r} is missing or still a placeholder")
            continue
        input_path = _resolve(root, value).resolve()
        resolved[key] = str(input_path)
        if not input_path.is_file():
            problems.append(f"plan input {key!r} is missing: {input_path}")
            continue
        hashes[key] = sha256_file(input_path)
        if input_path.suffix.lower() in {".csv", ".toml", ".json"}:
            text = input_path.read_text(encoding="utf-8", errors="replace").upper()
            if "REPLACE_ME" in text or "REPLACE_" in text:
                problems.append(f"plan input {key!r} still contains template placeholders")
    registry = resolved.get("asset_registry")
    if registry and Path(registry).is_file():
        problems.extend(validate_asset_registry(Path(registry)))
    report = {
        "ready": not problems,
        "plan": str(path),
        "plan_sha256": sha256_file(path),
        "resolved_inputs": resolved,
        "input_sha256": hashes,
        "problems": problems,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def run_full_rerun(
    plan_path: str | Path,
    output_directory: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    preflight = preflight_full_rerun(plan_path)
    if not preflight["ready"]:
        message = "Full rerun preflight failed:\n- " + "\n- ".join(preflight["problems"])
        raise ValueError(message)
    path = Path(plan_path).resolve()
    plan, root = _read_plan(path)
    inputs = plan["inputs"]
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Full-rerun output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight_report.json", preflight)
    started = time.perf_counter()
    stages: dict[str, Any] = {}

    def input_path(key: str) -> Path:
        return _resolve(root, inputs[key]).resolve()

    nsi_output = output / "nsi"
    nsi = run_batch_manifest(input_path("nsi_manifest"), nsi_output, config)
    stages["nsi"] = {"rows": int(len(nsi)), "output": str(nsi_output)}

    sensitivity_output = output / "sensitivity"
    sensitivity = run_sensitivity_manifest(
        input_path("sensitivity_manifest"), sensitivity_output, config
    )
    stages["sensitivity"] = {
        "rows": int(len(sensitivity)),
        "output": str(sensitivity_output),
    }

    garw_output = output / "garw"
    stages["garw"] = run_garw_table(
        input_path("garw_candidates"), garw_output, config
    )

    matching = plan.get("matching", {})
    matching_output = output / "matching"
    stages["matching"] = run_match_evaluation(
        input_path("matches"),
        matching_output,
        config,
        thresholds=tuple(float(value) for value in matching.get("thresholds", [1, 3, 5, 10])),
        primary_method=matching.get("primary_method"),
        baseline_methods=tuple(str(value) for value in matching.get("baseline_methods", [])),
        symmetric_homography=bool(matching.get("symmetric_homography", True)),
    )

    association = plan.get("association", {})
    association_output = output / "association.json"
    stages["association"] = run_association_analysis(
        input_path("association"),
        association_output,
        config,
        x_column=str(association.get("x_column", "NSI")),
        y_column=str(association.get("y_column", "robustness")),
        group_column=association.get("group_column", "scene"),
    )

    stability = plan.get("rank_stability", {})
    stability_output = output / "rank_stability.csv"
    stability_table = run_rank_stability_analysis(
        input_path("rank_variants"),
        stability_output,
        item_column=str(stability.get("item_column", "model")),
        variant_column=str(stability.get("variant_column", "variant")),
        score_column=str(stability.get("score_column", "NSI")),
        baseline_variant=str(stability.get("baseline_variant", "default")),
    )
    stages["rank_stability"] = {
        "rows": int(len(stability_table)),
        "output": str(stability_output),
    }

    figures_output = output / "figures"
    stages["figures"] = generate_figures(output, figures_output)
    report = {
        "status": "complete",
        "plan": str(path),
        "plan_sha256": preflight["plan_sha256"],
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "source_sha256": source_sha256(),
        "software_versions": software_versions(),
        "preflight": preflight,
        "stages": stages,
        "started_utc": preflight["checked_utc"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "full_rerun_report.json", report)
    return report
