from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .statistics import PairedBootstrapResult, paired_group_bootstrap


EPS = 1e-12
HOMOGRAPHY_COLUMNS = [f"h{row}{column}" for row in range(3) for column in range(3)]


def warp_points(points_xy: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape N x 2")
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape 3 x 3")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed = homogeneous @ matrix.T
    denominator = transformed[:, 2]
    valid = np.isfinite(transformed).all(axis=1) & (np.abs(denominator) > EPS)
    projected = np.divide(
        transformed[:, :2],
        denominator[:, None],
        out=np.full((points.shape[0], 2), np.nan),
        where=valid[:, None],
    )
    return projected, valid


def homography_reprojection_errors(
    source_xy: np.ndarray,
    destination_xy: np.ndarray,
    homography_source_to_destination: np.ndarray,
    *,
    symmetric: bool = False,
) -> np.ndarray:
    source = np.asarray(source_xy, dtype=np.float64)
    destination = np.asarray(destination_xy, dtype=np.float64)
    if source.shape != destination.shape:
        raise ValueError("source and destination points must be aligned")
    projected, valid_forward = warp_points(source, homography_source_to_destination)
    forward_squared = np.sum((projected - destination) ** 2, axis=1)
    if not symmetric:
        return np.where(valid_forward, np.sqrt(forward_squared), np.nan)
    inverse = np.linalg.inv(np.asarray(homography_source_to_destination, dtype=np.float64))
    back_projected, valid_backward = warp_points(destination, inverse)
    backward_squared = np.sum((back_projected - source) ** 2, axis=1)
    valid = valid_forward & valid_backward
    return np.where(valid, np.sqrt((forward_squared + backward_squared) / 2.0), np.nan)


def depth_pose_reprojection_errors(
    source_xy: np.ndarray,
    source_depth: np.ndarray,
    observed_destination_xy: np.ndarray,
    source_intrinsics: np.ndarray,
    destination_intrinsics: np.ndarray,
    transform_destination_from_source: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_xy, dtype=np.float64)
    observed = np.asarray(observed_destination_xy, dtype=np.float64)
    depth = np.asarray(source_depth, dtype=np.float64).reshape(-1)
    k_source = np.asarray(source_intrinsics, dtype=np.float64)
    k_destination = np.asarray(destination_intrinsics, dtype=np.float64)
    transform = np.asarray(transform_destination_from_source, dtype=np.float64)
    if source.shape != observed.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Point arrays must be aligned N x 2 matrices")
    if depth.shape[0] != source.shape[0]:
        raise ValueError("source_depth must be aligned with points")
    if k_source.shape != (3, 3) or k_destination.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Intrinsics must be 3x3 and pose transform must be 4x4")
    pixels = np.concatenate([source, np.ones((source.shape[0], 1))], axis=1)
    rays = pixels @ np.linalg.inv(k_source).T
    points_source = rays * depth[:, None]
    homogeneous = np.concatenate([points_source, np.ones((source.shape[0], 1))], axis=1)
    points_destination = homogeneous @ transform.T
    camera = points_destination[:, :3]
    projected_homogeneous = camera @ k_destination.T
    z = projected_homogeneous[:, 2]
    valid = (
        np.isfinite(projected_homogeneous).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (z > EPS)
    )
    projected = np.divide(
        projected_homogeneous[:, :2],
        z[:, None],
        out=np.full_like(observed, np.nan),
        where=valid[:, None],
    )
    error = np.linalg.norm(projected - observed, axis=1)
    return np.where(valid, error, np.nan)


def mma_curve(errors: np.ndarray | Iterable[float], thresholds: Iterable[float]) -> dict[float, float]:
    value = np.asarray(errors, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return {float(threshold): float("nan") for threshold in thresholds}
    return {
        float(threshold): float(np.mean(value <= float(threshold)))
        for threshold in thresholds
    }


@dataclass(frozen=True)
class MatchEvaluation:
    summary: dict[str, dict[str, float]]
    paired_tests: dict[str, PairedBootstrapResult]
    per_match: pd.DataFrame
    per_pair: pd.DataFrame
    per_scene: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "paired_tests": {
                name: value.to_dict() for name, value in self.paired_tests.items()
            },
        }


def _derive_errors(table: pd.DataFrame, symmetric: bool) -> np.ndarray:
    if "reprojection_error" in table.columns:
        return table["reprojection_error"].to_numpy(dtype=np.float64)
    coordinate_columns = {"x1", "y1", "x2", "y2"}
    missing = sorted(coordinate_columns.union(HOMOGRAPHY_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(
            "Match table needs reprojection_error or coordinates plus homography columns; "
            f"missing {missing}"
        )
    errors = np.full(len(table), np.nan, dtype=np.float64)
    positions = pd.Series(np.arange(len(table)), index=table.index)
    grouping = [column for column in ("method", "pair_id") if column in table.columns]
    for _, block in table.groupby(grouping, sort=False):
        matrices = block[HOMOGRAPHY_COLUMNS].drop_duplicates()
        if len(matrices) != 1:
            raise ValueError("Every method/pair block must use exactly one homography")
        matrix = matrices.iloc[0].to_numpy(dtype=np.float64).reshape(3, 3)
        block_errors = homography_reprojection_errors(
            block[["x1", "y1"]].to_numpy(dtype=np.float64),
            block[["x2", "y2"]].to_numpy(dtype=np.float64),
            matrix,
            symmetric=symmetric,
        )
        errors[positions.loc[block.index].to_numpy(dtype=int)] = block_errors
    return errors


def evaluate_match_table(
    input_table: pd.DataFrame,
    *,
    thresholds: Iterable[float] = (1.0, 3.0, 5.0, 10.0),
    primary_method: str | None = None,
    baseline_methods: Iterable[str] = (),
    symmetric_homography: bool = False,
    bootstrap_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> MatchEvaluation:
    table = input_table.copy().reset_index(drop=True)
    required = {"pair_id", "scene", "method"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Match table is missing columns: {missing}")
    threshold_values = tuple(sorted({float(value) for value in thresholds}))
    if not threshold_values or any(value <= 0.0 for value in threshold_values):
        raise ValueError("MMA thresholds must be positive")
    table["reprojection_error"] = _derive_errors(table, symmetric_homography)
    if not np.isfinite(table["reprojection_error"]).any():
        raise ValueError("No valid reprojection error could be computed")
    rows: list[dict[str, Any]] = []
    for (method, pair_id), block in table.groupby(["method", "pair_id"], sort=True):
        if block["scene"].astype(str).nunique() != 1:
            raise ValueError(f"Pair {pair_id!r} spans more than one scene")
        errors = block["reprojection_error"].to_numpy(dtype=np.float64)
        finite = errors[np.isfinite(errors)]
        row: dict[str, Any] = {
            "method": str(method),
            "pair_id": str(pair_id),
            "scene": str(block["scene"].iloc[0]),
            "match_count": int(finite.size),
            "median_reprojection_error": float(np.median(finite)) if finite.size else np.nan,
        }
        for threshold, value in mma_curve(finite, threshold_values).items():
            row[f"mma_at_{threshold:g}"] = value
        rows.append(row)
    per_pair = pd.DataFrame(rows)
    metric_columns = [f"mma_at_{threshold:g}" for threshold in threshold_values]
    per_scene = (
        per_pair.groupby(["scene", "method"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["method", "scene"])
        .reset_index(drop=True)
    )
    summary: dict[str, dict[str, float]] = {}
    for method, block in per_scene.groupby("method", sort=True):
        summary[str(method)] = {
            column: float(block[column].mean()) for column in metric_columns
        }
        summary[str(method)]["scene_count"] = float(block["scene"].nunique())
        summary[str(method)]["pair_count"] = float(
            per_pair.loc[per_pair["method"] == method, "pair_id"].nunique()
        )
    paired_tests: dict[str, PairedBootstrapResult] = {}
    if primary_method is not None:
        if primary_method not in set(per_pair["method"].astype(str)):
            raise ValueError(f"Primary method {primary_method!r} is absent")
        comparison_metric = metric_columns[-1]
        pivot = per_pair.pivot(index="pair_id", columns="method", values=comparison_metric)
        scene_lookup = per_pair.drop_duplicates("pair_id").set_index("pair_id")["scene"]
        for baseline in baseline_methods:
            if baseline not in pivot.columns:
                raise ValueError(f"Baseline method {baseline!r} is absent")
            aligned = pivot.dropna(subset=[primary_method, baseline])
            if aligned.empty:
                raise ValueError(f"No aligned pairs for {primary_method!r} and {baseline!r}")
            paired_tests[f"{primary_method}_vs_{baseline}_{comparison_metric}"] = paired_group_bootstrap(
                aligned[primary_method].to_numpy(),
                aligned[baseline].to_numpy(),
                scene_lookup.loc[aligned.index].to_numpy(),
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + len(paired_tests),
            )
    return MatchEvaluation(
        summary=summary,
        paired_tests=paired_tests,
        per_match=table,
        per_pair=per_pair,
        per_scene=per_scene,
    )
