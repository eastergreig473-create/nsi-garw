from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import FeatureDataset
from .metrics import NSIResult, estimate_information_purity, fit_ridge_probe, mean_finite, r2_per_output


EPS = 1e-12


@dataclass(frozen=True)
class LocalNSIProxyModel:
    feature_columns: tuple[str, ...]
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients_standardized: np.ndarray
    calibration_group_count: int
    fold_count: int
    validation_mse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != len(self.feature_columns):
            raise ValueError("Proxy feature matrix has the wrong shape")
        standardized = (value - self.x_mean) / self.x_scale
        prediction = standardized @ self.coefficients_standardized
        prediction = prediction * self.y_scale + self.y_mean
        return np.clip(prediction[:, 0], 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "alpha": self.alpha,
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_scale": self.y_scale.tolist(),
            "coefficients_standardized": self.coefficients_standardized.tolist(),
            "calibration_group_count": self.calibration_group_count,
            "fold_count": self.fold_count,
            "validation_mse": self.validation_mse,
            "output_range": [0.0, 1.0],
            "calibration_prediction_mode": "group-out-of-fold",
            "evaluation_prediction_mode": "fit-on-all-calibration-groups",
        }


def _group_folds(groups: np.ndarray, fold_count: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(groups.astype(str))
    if unique.size < 3:
        raise ValueError("At least three calibration groups are required for an NSI proxy")
    resolved_folds = min(max(2, int(fold_count)), unique.size)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    group_blocks = np.array_split(shuffled, resolved_folds)
    return [np.isin(groups.astype(str), block) for block in group_blocks]


def fit_local_nsi_proxy(
    table: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_column: str = "nsi_reference",
    group_column: str = "scene",
    split_column: str = "split",
    lambdas: Iterable[float] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0),
    fold_count: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, LocalNSIProxyModel, list[dict[str, float]]]:
    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("At least one observable proxy feature is required")
    required = set(columns).union({target_column, group_column, split_column})
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Local-NSI proxy table is missing columns: {missing}")
    split = table[split_column].astype(str).str.strip().str.lower().replace(
        {"val": "calibration", "validation": "calibration", "test": "evaluation"}
    )
    unknown = sorted(set(split).difference({"calibration", "evaluation"}))
    if unknown:
        raise ValueError(f"Unknown proxy split labels: {unknown}")
    calibration_mask = split == "calibration"
    evaluation_mask = split == "evaluation"
    if not calibration_mask.any() or not evaluation_mask.any():
        raise ValueError("Proxy table must contain calibration and evaluation rows")
    calibration_groups = set(table.loc[calibration_mask, group_column].astype(str))
    evaluation_groups = set(table.loc[evaluation_mask, group_column].astype(str))
    overlap = sorted(calibration_groups.intersection(evaluation_groups))
    if overlap:
        raise ValueError(f"Group leakage in local-NSI proxy table: {overlap[:5]}")
    x_calibration = table.loc[calibration_mask, list(columns)].to_numpy(dtype=np.float64)
    y_calibration = table.loc[calibration_mask, target_column].to_numpy(dtype=np.float64)
    groups = table.loc[calibration_mask, group_column].astype(str).to_numpy()
    if not np.isfinite(x_calibration).all() or not np.isfinite(y_calibration).all():
        raise ValueError("Calibration proxy features and targets must be finite")
    if np.any((y_calibration < 0.0) | (y_calibration > 1.0)):
        raise ValueError("nsi_reference must lie in [0, 1]")
    folds = _group_folds(groups, fold_count, seed)
    candidate_lambdas = tuple(sorted({float(value) for value in lambdas}))
    if not candidate_lambdas or any(value < 0.0 for value in candidate_lambdas):
        raise ValueError("Proxy ridge lambdas must be non-negative")
    traces: list[dict[str, float]] = []
    best_alpha = candidate_lambdas[0]
    best_mse = np.inf
    best_oof = np.zeros_like(y_calibration)
    for alpha in candidate_lambdas:
        oof = np.full_like(y_calibration, np.nan)
        for held_out in folds:
            fit = ~held_out
            model = fit_ridge_probe(x_calibration[fit], y_calibration[fit], alpha)
            oof[held_out] = np.clip(model.predict(x_calibration[held_out])[:, 0], 0.0, 1.0)
        mse = float(np.mean((oof - y_calibration) ** 2))
        traces.append({"alpha": alpha, "group_oof_mse": mse})
        if mse < best_mse - 1e-15:
            best_alpha, best_mse, best_oof = alpha, mse, oof
    final_probe = fit_ridge_probe(x_calibration, y_calibration, best_alpha)
    model = LocalNSIProxyModel(
        feature_columns=columns,
        alpha=best_alpha,
        x_mean=final_probe.x_mean,
        x_scale=final_probe.x_scale,
        y_mean=final_probe.y_mean,
        y_scale=final_probe.y_scale,
        coefficients_standardized=final_probe.coefficients_standardized,
        calibration_group_count=len(calibration_groups),
        fold_count=len(folds),
        validation_mse=best_mse,
    )
    output = table.copy()
    output["nsi_local"] = np.nan
    output.loc[calibration_mask, "nsi_local"] = best_oof
    x_evaluation = table.loc[evaluation_mask, list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(x_evaluation).all():
        raise ValueError("Evaluation proxy features must be finite")
    output.loc[evaluation_mask, "nsi_local"] = model.predict(x_evaluation)
    output["nsi_mode"] = "non_oracle_proxy"
    return output, model, traces


def compute_local_nsi_references(
    dataset: FeatureDataset,
    global_result: NSIResult,
    window_ids: np.ndarray,
    config: PipelineConfig,
    *,
    eligible_mask: np.ndarray | None = None,
    minimum_samples: int | None = None,
) -> pd.DataFrame:
    """Compute annotation-dependent local NSI targets for proxy calibration only.

    Probes and subspaces stay fixed. Accessibility and dependence are evaluated
    inside each held-out local window. These values are oracle/reference targets
    and must not be passed directly to the non-oracle evaluation gate.
    """
    dataset.validate()
    windows = np.asarray(window_ids).astype(str).reshape(-1)
    if windows.shape[0] != dataset.features.shape[0]:
        raise ValueError("window_ids must be aligned with the feature dataset")
    if eligible_mask is None:
        eligible = np.ones(windows.shape[0], dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if eligible.shape != windows.shape:
            raise ValueError("eligible_mask must be aligned with window_ids")
    required_samples = max(config.ksg.k + 2, 12) if minimum_samples is None else int(minimum_samples)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(np.unique(windows[eligible])):
        mask = eligible & (windows == window)
        sample_count = int(mask.sum())
        if sample_count < required_samples:
            continue
        geometry_prediction = global_result.geometry_probe.predict(dataset.features[mask])
        appearance_prediction = global_result.appearance_probe.predict(dataset.features[mask])
        geometry_r2 = mean_finite(r2_per_output(dataset.geometry[mask], geometry_prediction))
        appearance_r2 = mean_finite(r2_per_output(dataset.appearance[mask], appearance_prediction))
        accessibility = float(
            (np.clip(geometry_r2, 0.0, 1.0) + np.clip(appearance_r2, 0.0, 1.0)) / 2.0
        )
        local_config = PipelineConfig(
            seed=config.seed + index + 1,
            split=config.split,
            ridge=config.ridge,
            subspace=config.subspace,
            ksg=config.ksg,
            statistics=config.statistics,
            garw=config.garw,
            degradation=config.degradation,
        )
        dependence = estimate_information_purity(
            geometry_prediction, appearance_prediction, local_config
        )
        nsi_reference = accessibility * global_result.separation * dependence.purity
        rows.append(
            {
                "window_id": window,
                "sample_count": sample_count,
                "A_accessibility": accessibility,
                "B_separation_global": global_result.separation,
                "C_information_purity": dependence.purity,
                "nsi_reference": nsi_reference,
                "geometry_r2": geometry_r2,
                "appearance_r2": appearance_r2,
                "reference_mode": "annotation_oracle_proxy_target_only",
            }
        )
    return pd.DataFrame(rows)
