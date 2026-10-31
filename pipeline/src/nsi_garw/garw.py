from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from .statistics import PairedBootstrapResult, paired_group_bootstrap


EPS = 1e-12
REQUIRED_COLUMNS = {
    "query_id",
    "scene",
    "label",
    "s_geometry",
    "s_appearance",
    "nsi_local",
    "nsi_mode",
}


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def row_normalize(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    lengths = np.linalg.norm(value, axis=1, keepdims=True)
    return np.divide(value, lengths, out=np.zeros_like(value), where=lengths > EPS)


def projected_cosine(
    first_features: np.ndarray,
    second_features: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first_features, dtype=np.float64)
    second = np.asarray(second_features, dtype=np.float64)
    projector = np.asarray(basis, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Feature arrays must be aligned two-dimensional matrices")
    if projector.ndim != 2 or projector.shape[0] != first.shape[1]:
        raise ValueError("basis must have shape feature_dimension x rank")
    if projector.shape[1] == 0:
        return np.zeros(first.shape[0], dtype=np.float64)
    first_projected = row_normalize(first @ projector)
    second_projected = row_normalize(second @ projector)
    return np.clip(np.sum(first_projected * second_projected, axis=1), -1.0, 1.0)


def standardized_projected_cosine(
    first_features: np.ndarray,
    second_features: np.ndarray,
    basis: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    """Project using bases fitted in standardized feature coordinates."""
    first = np.asarray(first_features, dtype=np.float64)
    second = np.asarray(second_features, dtype=np.float64)
    mean = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("Feature arrays must be aligned two-dimensional matrices")
    if mean.shape != (first.shape[1],) or scale.shape != (first.shape[1],):
        raise ValueError("Normalization statistics do not match the feature dimension")
    safe_scale = np.where(scale > EPS, scale, 1.0)
    return projected_cosine(
        (first - mean) / safe_scale,
        (second - mean) / safe_scale,
        basis,
    )


def garw_scores(
    geometry_similarity: np.ndarray | Iterable[float],
    appearance_similarity: np.ndarray | Iterable[float],
    nsi_local: np.ndarray | Iterable[float],
    *,
    alpha: float,
    beta: float,
    penalty_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = np.asarray(geometry_similarity, dtype=np.float64).reshape(-1)
    appearance = np.asarray(appearance_similarity, dtype=np.float64).reshape(-1)
    local = np.asarray(nsi_local, dtype=np.float64).reshape(-1)
    if geometry.shape != appearance.shape or geometry.shape != local.shape:
        raise ValueError("Similarities and local NSI must be aligned")
    if penalty_lambda < 0.0:
        raise ValueError("penalty_lambda must be non-negative")
    if not (
        np.isfinite(geometry).all()
        and np.isfinite(appearance).all()
        and np.isfinite(local).all()
    ):
        raise ValueError("GARW inputs must be finite")
    gate = sigmoid(float(alpha) * local + float(beta))
    unpenalized = (1.0 - gate) * geometry + gate * appearance
    conflict = (
        float(penalty_lambda)
        * 4.0
        * gate
        * (1.0 - gate)
        * np.abs(geometry - appearance)
    )
    return unpenalized - conflict, gate, conflict


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != value.shape:
        raise ValueError("labels and scores must be aligned")
    positive = target == 1
    negative = target == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(value, method="average")
    rank_sum = float(ranks[positive].sum())
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int((target == 1).sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-value, kind="mergesort")
    ordered = target[order] == 1
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(precision[ordered].mean())


def validate_candidate_table(table: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"GARW candidate table is missing columns: {missing}")
    if table.empty:
        raise ValueError("GARW candidate table is empty")
    numeric = table[["label", "s_geometry", "s_appearance", "nsi_local"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("GARW candidate table contains non-finite values")
    raw_labels = table["label"].to_numpy(dtype=np.float64)
    if not np.isin(raw_labels, [0.0, 1.0]).all() or set(raw_labels.tolist()) != {0.0, 1.0}:
        raise ValueError("GARW labels must contain both binary classes 0 and 1")
    local = table["nsi_local"].to_numpy(dtype=np.float64)
    if np.any((local < 0.0) | (local > 1.0)):
        raise ValueError("nsi_local must lie in [0, 1]")
    modes = set(table["nsi_mode"].astype(str))
    permitted_modes = {"non_oracle_proxy", "oracle_annotation"}
    if not modes or not modes.issubset(permitted_modes):
        raise ValueError(f"nsi_mode must use one of {sorted(permitted_modes)}")
    if len(modes) != 1:
        raise ValueError("Run non-oracle and oracle NSI modes as separate GARW analyses")
    for query_id, block in table.groupby("query_id", sort=False):
        if block["scene"].astype(str).nunique() != 1:
            raise ValueError(f"Query {query_id!r} spans more than one scene")


@dataclass(frozen=True)
class GARWParameters:
    alpha: float
    beta: float
    penalty_lambda: float
    logit_scale: float
    logit_bias: float
    calibration_loss: float
    calibration_rows: int
    calibration_queries: int
    calibration_scenes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "penalty_lambda": self.penalty_lambda,
            "logit_scale": self.logit_scale,
            "logit_bias": self.logit_bias,
            "calibration_loss": self.calibration_loss,
            "calibration_rows": self.calibration_rows,
            "calibration_queries": self.calibration_queries,
            "calibration_scenes": self.calibration_scenes,
        }


def _binary_cross_entropy(
    labels: np.ndarray, logits: np.ndarray, weights: np.ndarray | None = None
) -> float:
    # Stable logistic loss: max(z,0)-y*z+log(1+exp(-abs(z))).
    loss = np.maximum(logits, 0.0) - labels * logits + np.log1p(
        np.exp(-np.abs(logits))
    )
    if weights is None:
        return float(loss.mean())
    normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
    if normalized.shape != loss.shape or np.any(normalized < 0.0):
        raise ValueError("Calibration weights must be non-negative and aligned")
    if normalized.sum() <= 0.0:
        raise ValueError("Calibration weights must have positive mass")
    return float(np.average(loss, weights=normalized))


def calibrate_garw(
    calibration_table: pd.DataFrame,
    penalty_lambdas: Iterable[float],
    *,
    seed: int = 0,
) -> tuple[GARWParameters, list[dict[str, float]]]:
    validate_candidate_table(calibration_table)
    table = calibration_table.copy()
    labels = table["label"].to_numpy(dtype=np.float64)
    geometry = table["s_geometry"].to_numpy(dtype=np.float64)
    appearance = table["s_appearance"].to_numpy(dtype=np.float64)
    local = table["nsi_local"].to_numpy(dtype=np.float64)
    # Give every scene equal mass, every query equal mass within a scene, and
    # every candidate equal mass within a query. This prevents large scenes or
    # dense candidate lists from dominating cross-scene calibration.
    scene_query_counts = table.groupby("scene")["query_id"].transform("nunique").to_numpy(dtype=np.float64)
    query_candidate_counts = table.groupby("query_id")["query_id"].transform("size").to_numpy(dtype=np.float64)
    scene_count = float(table["scene"].astype(str).nunique())
    calibration_weights = 1.0 / (scene_count * scene_query_counts * query_candidate_counts)
    candidates = tuple(sorted({float(value) for value in penalty_lambdas}))
    if not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("penalty_lambdas must contain non-negative values")
    rng = np.random.default_rng(seed)
    traces: list[dict[str, float]] = []
    best: GARWParameters | None = None

    for penalty_lambda in candidates:
        def objective(parameters: np.ndarray) -> float:
            alpha, beta, log_scale, intercept = parameters
            fused, _, _ = garw_scores(
                geometry,
                appearance,
                local,
                alpha=float(alpha),
                beta=float(beta),
                penalty_lambda=penalty_lambda,
            )
            logits = np.exp(log_scale) * fused + intercept
            regularization = 1e-5 * float(alpha**2 + beta**2 + log_scale**2)
            return _binary_cross_entropy(labels, logits, calibration_weights) + regularization

        starting_points = [
            np.asarray([1.0, 0.0, np.log(5.0), 0.0]),
            np.asarray([-1.0, 0.0, np.log(5.0), 0.0]),
        ]
        starting_points.extend(
            rng.normal(
                loc=np.asarray([0.0, 0.0, np.log(5.0), 0.0]),
                scale=np.asarray([2.0, 1.0, 0.5, 0.5]),
                size=(2, 4),
            )
        )
        best_result = None
        for start in starting_points:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=[(-20.0, 20.0), (-10.0, 10.0), (-5.0, 5.0), (-10.0, 10.0)],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
        assert best_result is not None
        alpha, beta, log_scale, intercept = (
            float(value) for value in best_result.x
        )
        candidate = GARWParameters(
            alpha=alpha,
            beta=beta,
            penalty_lambda=penalty_lambda,
            logit_scale=float(np.exp(log_scale)),
            logit_bias=intercept,
            calibration_loss=float(best_result.fun),
            calibration_rows=int(len(table)),
            calibration_queries=int(table["query_id"].astype(str).nunique()),
            calibration_scenes=int(table["scene"].astype(str).nunique()),
        )
        traces.append(
            {
                "penalty_lambda": penalty_lambda,
                "calibration_loss": candidate.calibration_loss,
                "alpha": candidate.alpha,
                "beta": candidate.beta,
                "logit_scale": candidate.logit_scale,
                "logit_bias": candidate.logit_bias,
                "converged": float(bool(best_result.success)),
            }
        )
        if best is None or candidate.calibration_loss < best.calibration_loss - 1e-12:
            best = candidate
    assert best is not None
    return best, traces


def _query_metrics(table: pd.DataFrame, score_column: str, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, block in table.groupby("query_id", sort=True):
        labels = block["label"].to_numpy(dtype=np.int64)
        scores = block[score_column].to_numpy(dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        positive_positions = np.flatnonzero(labels[order] == 1)
        first_position = int(positive_positions[0]) if positive_positions.size else None
        rows.append(
            {
                "query_id": str(query_id),
                "scene": str(block["scene"].iloc[0]),
                "method": method,
                "top1": float(first_position == 0),
                "mrr": float(1.0 / (first_position + 1)) if first_position is not None else 0.0,
                "average_precision": average_precision(labels, scores) if positive_positions.size else 0.0,
                "candidate_auc": binary_auc(labels, scores),
                "positive_candidate_present": float(positive_positions.size > 0),
            }
        )
    return pd.DataFrame(rows)


def _oracle_branch_scores(table: pd.DataFrame) -> np.ndarray:
    scores = np.empty(len(table), dtype=np.float64)
    position = pd.Series(np.arange(len(table)), index=table.index)
    for _, block in table.groupby("query_id", sort=False):
        labels = block["label"].to_numpy(dtype=np.int64)
        geometry = block["s_geometry"].to_numpy(dtype=np.float64)
        appearance = block["s_appearance"].to_numpy(dtype=np.float64)
        chosen = geometry if average_precision(labels, geometry) >= average_precision(labels, appearance) else appearance
        scores[position.loc[block.index].to_numpy(dtype=int)] = chosen
    return scores


@dataclass(frozen=True)
class GARWEvaluation:
    parameters: GARWParameters
    summary: dict[str, dict[str, float]]
    paired_tests: dict[str, PairedBootstrapResult]
    per_query: pd.DataFrame
    per_scene: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters.to_dict(),
            "summary": self.summary,
            "paired_tests": {
                name: result.to_dict() for name, result in self.paired_tests.items()
            },
        }


def evaluate_garw(
    evaluation_table: pd.DataFrame,
    parameters: GARWParameters,
    *,
    bootstrap_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 0,
    lambda_zero_parameters: GARWParameters | None = None,
) -> GARWEvaluation:
    validate_candidate_table(evaluation_table)
    table = evaluation_table.copy().reset_index(drop=True)
    fused, gate, conflict = garw_scores(
        table["s_geometry"].to_numpy(),
        table["s_appearance"].to_numpy(),
        table["nsi_local"].to_numpy(),
        alpha=parameters.alpha,
        beta=parameters.beta,
        penalty_lambda=parameters.penalty_lambda,
    )
    zero_parameters = lambda_zero_parameters or parameters
    no_penalty, _, _ = garw_scores(
        table["s_geometry"].to_numpy(),
        table["s_appearance"].to_numpy(),
        table["nsi_local"].to_numpy(),
        alpha=zero_parameters.alpha,
        beta=zero_parameters.beta,
        penalty_lambda=0.0,
    )
    table["garw_score"] = fused
    table["garw_no_penalty_score"] = no_penalty
    table["gate"] = gate
    table["conflict_penalty"] = conflict
    table["oracle_branch_score"] = _oracle_branch_scores(table)
    method_columns = {
        "geometry": "s_geometry",
        "appearance": "s_appearance",
        "garw": "garw_score",
        "garw_lambda_zero": "garw_no_penalty_score",
        "oracle_branch": "oracle_branch_score",
    }
    table["fixed_weight_score"] = 0.5 * (table["s_geometry"] + table["s_appearance"])
    method_columns["fixed_weight"] = "fixed_weight_score"
    if "s_baseline" in table.columns:
        method_columns["descriptor_baseline"] = "s_baseline"
    blocks = [
        _query_metrics(table, score_column, method)
        for method, score_column in method_columns.items()
    ]
    per_query = pd.concat(blocks, ignore_index=True)
    metric_names = ["top1", "mrr", "average_precision", "candidate_auc", "positive_candidate_present"]
    summary: dict[str, dict[str, float]] = {}
    for method, block in per_query.groupby("method", sort=True):
        summary[str(method)] = {
            metric: float(block[metric].mean()) for metric in metric_names
        }
        summary[str(method)]["query_count"] = float(len(block))
    per_scene = (
        per_query.groupby(["scene", "method"], as_index=False)[metric_names]
        .mean()
        .sort_values(["method", "scene"])
        .reset_index(drop=True)
    )
    for method, block in per_scene.groupby("method", sort=True):
        summary[str(method)].update({metric: float(block[metric].mean()) for metric in metric_names})
        summary[str(method)]["scene_count"] = float(len(block))
    pivot = per_query.pivot(index="query_id", columns="method", values="average_precision")
    scene_lookup = per_query.drop_duplicates("query_id").set_index("query_id")["scene"]
    pivot = pivot.dropna(subset=["garw", "geometry", "appearance", "garw_lambda_zero"])
    paired_tests = {
        "garw_vs_geometry": paired_group_bootstrap(
            pivot["garw"].to_numpy(),
            pivot["geometry"].to_numpy(),
            scene_lookup.loc[pivot.index].to_numpy(),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
        ),
        "garw_vs_appearance": paired_group_bootstrap(
            pivot["garw"].to_numpy(),
            pivot["appearance"].to_numpy(),
            scene_lookup.loc[pivot.index].to_numpy(),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + 1,
        ),
        "penalty_vs_lambda_zero": paired_group_bootstrap(
            pivot["garw"].to_numpy(),
            pivot["garw_lambda_zero"].to_numpy(),
            scene_lookup.loc[pivot.index].to_numpy(),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + 2,
        ),
    }
    for metric in ("average_precision", "top1"):
        paired = per_query.pivot(index="query_id", columns="method", values=metric)
        for baseline in ("descriptor_baseline", "fixed_weight"):
            if baseline in paired:
                paired_tests[f"garw_vs_{baseline}_{metric}"] = paired_group_bootstrap(
                    paired["garw"].to_numpy(), paired[baseline].to_numpy(),
                    scene_lookup.loc[paired.index].to_numpy(),
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=confidence_level, seed=seed + 3,
                )
    return GARWEvaluation(
        parameters=parameters,
        summary=summary,
        paired_tests=paired_tests,
        per_query=per_query,
        per_scene=per_scene,
    )


def split_calibration_evaluation(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in table.columns:
        raise ValueError("GARW table requires a split column")
    labels = table["split"].astype(str).str.strip().str.lower().replace(
        {"val": "calibration", "validation": "calibration", "test": "evaluation"}
    )
    unknown = sorted(set(labels).difference({"calibration", "evaluation"}))
    if unknown:
        raise ValueError(f"Unknown GARW split labels: {unknown}")
    calibration = table[labels == "calibration"].copy()
    evaluation = table[labels == "evaluation"].copy()
    if calibration.empty or evaluation.empty:
        raise ValueError("split must contain calibration and evaluation rows")
    calibration_scenes = set(calibration["scene"].astype(str))
    evaluation_scenes = set(evaluation["scene"].astype(str))
    overlap = sorted(calibration_scenes.intersection(evaluation_scenes))
    if overlap:
        raise ValueError(f"Scene leakage between calibration and evaluation: {overlap[:5]}")
    validate_candidate_table(calibration)
    validate_candidate_table(evaluation)
    return calibration, evaluation
