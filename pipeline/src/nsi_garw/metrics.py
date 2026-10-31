from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from .config import PipelineConfig
from .io import FeatureDataset, canonical_split_array, deterministic_group_split


EPS = 1e-12


def _as_2d(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional array, received {value.shape}")
    return value


@dataclass(frozen=True)
class RidgeProbe:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients_standardized: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        standardized = (x - self.x_mean) / self.x_scale
        prediction_standardized = standardized @ self.coefficients_standardized
        return prediction_standardized * self.y_scale + self.y_mean


def fit_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    *,
    x_mean: np.ndarray | None = None,
    x_scale: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_scale: np.ndarray | None = None,
) -> RidgeProbe:
    x = _as_2d(features)
    y = _as_2d(targets)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Features and targets must have the same number of rows")
    resolved_x_mean = x.mean(axis=0) if x_mean is None else np.asarray(x_mean, dtype=np.float64)
    resolved_x_scale = x.std(axis=0, ddof=0) if x_scale is None else np.asarray(x_scale, dtype=np.float64)
    resolved_y_mean = y.mean(axis=0) if y_mean is None else np.asarray(y_mean, dtype=np.float64)
    resolved_y_scale = y.std(axis=0, ddof=0) if y_scale is None else np.asarray(y_scale, dtype=np.float64)
    if resolved_x_mean.shape != (x.shape[1],) or resolved_x_scale.shape != (x.shape[1],):
        raise ValueError("Supplied feature normalization statistics have the wrong shape")
    if resolved_y_mean.shape != (y.shape[1],) or resolved_y_scale.shape != (y.shape[1],):
        raise ValueError("Supplied target normalization statistics have the wrong shape")
    resolved_x_scale = np.where(resolved_x_scale > EPS, resolved_x_scale, 1.0)
    resolved_y_scale = np.where(resolved_y_scale > EPS, resolved_y_scale, 1.0)
    xs = (x - resolved_x_mean) / resolved_x_scale
    ys = (y - resolved_y_mean) / resolved_y_scale
    gram = xs.T @ xs
    regularized = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    rhs = xs.T @ ys
    try:
        coefficients = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(regularized, rcond=1e-12) @ rhs
    return RidgeProbe(
        x_mean=resolved_x_mean,
        x_scale=resolved_x_scale,
        y_mean=resolved_y_mean,
        y_scale=resolved_y_scale,
        coefficients_standardized=coefficients,
        alpha=float(alpha),
    )


def r2_per_output(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    y = _as_2d(actual)
    y_hat = _as_2d(predicted)
    if y.shape != y_hat.shape:
        raise ValueError("Actual and predicted arrays must have equal shape")
    residual = np.sum((y - y_hat) ** 2, axis=0)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    score = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(residual, total, out=score, where=total > EPS)
    score[total > EPS] = 1.0 - score[total > EPS]
    return score


def mean_finite(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float(fallback)


def select_ridge_alpha(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    lambdas: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    x_train = _as_2d(train_features)
    y_train = _as_2d(train_targets)
    x_validation = _as_2d(validation_features)
    y_validation = _as_2d(validation_targets)
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training features and targets must have the same number of rows")
    if x_validation.shape[0] != y_validation.shape[0]:
        raise ValueError("Validation features and targets must have the same number of rows")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Training and validation feature dimensions must match")

    # All candidate lambdas share the same standardized design matrix.  Build
    # and diagonalize its Gram matrix once instead of repeating the expensive
    # X^T X construction and dense solve for every candidate.
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0, ddof=0)
    x_scale = np.where(x_scale > EPS, x_scale, 1.0)
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0, ddof=0)
    y_scale = np.where(y_scale > EPS, y_scale, 1.0)
    xs = (x_train - x_mean) / x_scale
    ys = (y_train - y_mean) / y_scale
    validation_standardized = (x_validation - x_mean) / x_scale
    gram = xs.T @ xs
    rhs = xs.T @ ys
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    transformed_rhs = eigenvectors.T @ rhs

    records: list[dict[str, float]] = []
    best_alpha = float(lambdas[0])
    best_score = -np.inf
    for alpha in sorted(float(value) for value in lambdas):
        denominator = eigenvalues + alpha
        inverse = np.divide(
            1.0,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > EPS,
        )
        coefficients = eigenvectors @ (inverse[:, None] * transformed_rhs)
        prediction = (
            validation_standardized @ coefficients
        ) * y_scale + y_mean
        score = mean_finite(
            r2_per_output(y_validation, prediction),
            fallback=-np.inf,
        )
        records.append({"alpha": alpha, "validation_r2": score})
        if score > best_score + 1e-15:
            best_score = score
            best_alpha = alpha
    return best_alpha, records


def orthonormal_basis(
    directions: np.ndarray,
    max_rank: int,
    relative_tolerance: float,
) -> np.ndarray:
    matrix = _as_2d(directions)
    if matrix.size == 0 or np.linalg.norm(matrix) <= EPS:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= EPS:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    numerical_rank = int(
        np.sum(singular_values > singular_values[0] * float(relative_tolerance))
    )
    rank = min(int(max_rank), numerical_rank)
    return u[:, :rank]


def normalized_subspace_overlap(
    first: np.ndarray,
    second: np.ndarray,
    metric: str = "mean_squared_principal_cosine",
) -> float:
    u = _as_2d(first)
    v = _as_2d(second)
    if u.shape[0] != v.shape[0]:
        raise ValueError("Subspace bases must use the same feature coordinates")
    denominator = min(u.shape[1], v.shape[1])
    if denominator == 0:
        return 1.0
    squared_cosines = np.linalg.svd(u.T @ v, compute_uv=False) ** 2
    if metric == "mean_squared_principal_cosine":
        overlap = float(squared_cosines.sum() / denominator)
    elif metric == "maximum_squared_principal_cosine":
        overlap = float(squared_cosines.max(initial=0.0))
    else:
        raise ValueError(f"Unsupported subspace-overlap metric: {metric}")
    return float(np.clip(overlap, 0.0, 1.0))


def residualized_appearance_basis(
    geometry_basis: np.ndarray,
    appearance_directions: np.ndarray,
    max_rank: int,
    relative_tolerance: float,
) -> np.ndarray:
    geometry = _as_2d(geometry_basis)
    appearance = _as_2d(appearance_directions)
    if geometry.shape[1] == 0:
        residual = appearance
    else:
        residual = appearance - geometry @ (geometry.T @ appearance)
    return orthonormal_basis(residual, max_rank, relative_tolerance)


def _standardize_nonconstant(array: np.ndarray) -> np.ndarray:
    value = _as_2d(array)
    scale = value.std(axis=0, ddof=0)
    keep = scale > EPS
    if not np.any(keep):
        return np.zeros((value.shape[0], 0), dtype=np.float64)
    selected = value[:, keep]
    return (selected - selected.mean(axis=0)) / selected.std(axis=0, ddof=0)


def ksg_mutual_information(
    first: np.ndarray,
    second: np.ndarray,
    *,
    k: int = 5,
    jitter: float = 1e-10,
    seed: int = 0,
) -> float:
    """KSG-1 mutual information estimate in natural-log units."""
    x = _standardize_nonconstant(first)
    y = _standardize_nonconstant(second)
    if x.shape[0] != y.shape[0]:
        raise ValueError("KSG inputs must contain the same number of samples")
    n = x.shape[0]
    if x.shape[1] == 0 or y.shape[1] == 0:
        return 0.0
    if n <= k + 1:
        raise ValueError(f"KSG requires more than k+1 samples; received n={n}, k={k}")
    rng = np.random.default_rng(seed)
    if jitter > 0:
        x = x + rng.normal(scale=float(jitter), size=x.shape)
        y = y + rng.normal(scale=float(jitter), size=y.shape)
    joint = np.concatenate([x, y], axis=1)
    joint_tree = cKDTree(joint)
    distances = joint_tree.query(joint, k=k + 1, p=np.inf)[0][:, -1]
    radii = np.nextafter(distances, 0.0)
    x_tree = cKDTree(x)
    y_tree = cKDTree(y)
    nx = np.asarray(
        x_tree.query_ball_point(x, radii, p=np.inf, return_length=True),
        dtype=np.int64,
    ) - 1
    ny = np.asarray(
        y_tree.query_ball_point(y, radii, p=np.inf, return_length=True),
        dtype=np.int64,
    ) - 1
    estimate = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(estimate)


@dataclass(frozen=True)
class DependenceResult:
    observed_mi_nats: float
    permutation_median_mi_nats: float
    excess_mi_nats: float
    permutation_p_value: float
    purity: float
    sample_count: int


def estimate_information_purity(
    geometry_predictions: np.ndarray,
    appearance_predictions: np.ndarray,
    config: PipelineConfig,
) -> DependenceResult:
    x = _as_2d(geometry_predictions)
    y = _as_2d(appearance_predictions)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Prediction arrays must be aligned")
    rng = np.random.default_rng(config.seed)
    if x.shape[0] > config.ksg.max_samples:
        selected = np.sort(
            rng.choice(x.shape[0], size=config.ksg.max_samples, replace=False)
        )
        x = x[selected]
        y = y[selected]
    observed = ksg_mutual_information(
        x,
        y,
        k=config.ksg.k,
        jitter=config.ksg.jitter,
        seed=config.seed,
    )
    null_values = np.empty(config.ksg.permutations, dtype=np.float64)
    for index in range(config.ksg.permutations):
        permuted = y[rng.permutation(y.shape[0])]
        null_values[index] = ksg_mutual_information(
            x,
            permuted,
            k=config.ksg.k,
            jitter=config.ksg.jitter,
            seed=config.seed + index + 1,
        )
    null_median = float(np.median(null_values)) if null_values.size else 0.0
    excess = max(float(observed - null_median), 0.0)
    if config.ksg.purity_transform == "reciprocal_excess_nats":
        purity = float(1.0 / (1.0 + excess))
    elif config.ksg.purity_transform == "exp_negative_excess_nats":
        purity = float(np.exp(-excess))
    else:
        raise ValueError("Unsupported purity transform: " + config.ksg.purity_transform)
    p_value = float(
        (1 + np.sum(null_values >= observed)) / (null_values.size + 1)
    )
    return DependenceResult(
        observed_mi_nats=float(observed),
        permutation_median_mi_nats=null_median,
        excess_mi_nats=excess,
        permutation_p_value=p_value,
        purity=purity,
        sample_count=int(x.shape[0]),
    )


@dataclass(frozen=True)
class NSIResult:
    accessibility: float
    separation: float
    information_purity: float
    nsi: float
    geometry_r2: float
    appearance_r2: float
    geometry_r2_per_output: np.ndarray
    appearance_r2_per_output: np.ndarray
    osr: float
    osr_metric: str
    purity_transform: str
    dependence: DependenceResult
    geometry_alpha: float
    appearance_alpha: float
    geometry_rank: int
    appearance_raw_rank: int
    appearance_projected_rank: int
    split_counts: dict[str, int]
    split_group_counts: dict[str, int]
    geometry_basis: np.ndarray
    appearance_raw_basis: np.ndarray
    appearance_projected_basis: np.ndarray
    geometry_probe: RidgeProbe
    appearance_probe: RidgeProbe
    validation_trace: dict[str, list[dict[str, float]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "A_accessibility": self.accessibility,
            "B_separation": self.separation,
            "C_information_purity": self.information_purity,
            "NSI": self.nsi,
            "geometry_r2": self.geometry_r2,
            "appearance_r2": self.appearance_r2,
            "geometry_r2_per_output": self.geometry_r2_per_output.tolist(),
            "appearance_r2_per_output": self.appearance_r2_per_output.tolist(),
            "OSR": self.osr,
            "OSR_metric": self.osr_metric,
            "purity_transform": self.purity_transform,
            "ksg_observed_mi_nats": self.dependence.observed_mi_nats,
            "ksg_permutation_median_mi_nats": self.dependence.permutation_median_mi_nats,
            "ksg_excess_mi_nats": self.dependence.excess_mi_nats,
            "ksg_permutation_p_value": self.dependence.permutation_p_value,
            "ksg_sample_count": self.dependence.sample_count,
            "geometry_alpha": self.geometry_alpha,
            "appearance_alpha": self.appearance_alpha,
            "geometry_rank": self.geometry_rank,
            "appearance_raw_rank": self.appearance_raw_rank,
            "appearance_projected_rank": self.appearance_projected_rank,
            "split_counts": self.split_counts,
            "split_group_counts": self.split_group_counts,
            "validation_trace": self.validation_trace,
        }


def compute_nsi(dataset: FeatureDataset, config: PipelineConfig) -> NSIResult:
    dataset.validate()
    split = (
        canonical_split_array(dataset.split)
        if dataset.split is not None
        else deterministic_group_split(dataset.groups, config.split, config.seed)
    )
    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"
    geometry_alpha, geometry_trace = select_ridge_alpha(
        dataset.features[train_mask],
        dataset.geometry[train_mask],
        dataset.features[validation_mask],
        dataset.geometry[validation_mask],
        config.ridge.lambdas,
    )
    appearance_alpha, appearance_trace = select_ridge_alpha(
        dataset.features[train_mask],
        dataset.appearance[train_mask],
        dataset.features[validation_mask],
        dataset.appearance[validation_mask],
        config.ridge.lambdas,
    )
    fit_mask = train_mask | validation_mask
    training_features = dataset.features[train_mask]
    feature_mean = training_features.mean(axis=0)
    feature_scale = training_features.std(axis=0, ddof=0)
    geometry_training_targets = dataset.geometry[train_mask]
    appearance_training_targets = dataset.appearance[train_mask]
    geometry_probe = fit_ridge_probe(
        dataset.features[fit_mask],
        dataset.geometry[fit_mask],
        geometry_alpha,
        x_mean=feature_mean,
        x_scale=feature_scale,
        y_mean=geometry_training_targets.mean(axis=0),
        y_scale=geometry_training_targets.std(axis=0, ddof=0),
    )
    appearance_probe = fit_ridge_probe(
        dataset.features[fit_mask],
        dataset.appearance[fit_mask],
        appearance_alpha,
        x_mean=feature_mean,
        x_scale=feature_scale,
        y_mean=appearance_training_targets.mean(axis=0),
        y_scale=appearance_training_targets.std(axis=0, ddof=0),
    )
    geometry_prediction = geometry_probe.predict(dataset.features[test_mask])
    appearance_prediction = appearance_probe.predict(dataset.features[test_mask])
    geometry_r2_values = r2_per_output(dataset.geometry[test_mask], geometry_prediction)
    appearance_r2_values = r2_per_output(
        dataset.appearance[test_mask], appearance_prediction
    )
    geometry_r2 = mean_finite(geometry_r2_values)
    appearance_r2 = mean_finite(appearance_r2_values)
    accessibility = float(
        (
            np.clip(geometry_r2, 0.0, 1.0)
            + np.clip(appearance_r2, 0.0, 1.0)
        )
        / 2.0
    )
    geometry_directions = geometry_probe.coefficients_standardized
    appearance_directions = appearance_probe.coefficients_standardized
    geometry_basis = orthonormal_basis(
        geometry_directions,
        config.subspace.max_rank,
        config.subspace.relative_singular_value_tolerance,
    )
    appearance_raw_basis = orthonormal_basis(
        appearance_directions,
        config.subspace.max_rank,
        config.subspace.relative_singular_value_tolerance,
    )
    osr = normalized_subspace_overlap(
        geometry_basis, appearance_raw_basis, config.subspace.overlap_metric
    )
    separation = float(np.clip(1.0 - osr, 0.0, 1.0))
    appearance_projected_basis = residualized_appearance_basis(
        geometry_basis,
        appearance_directions,
        config.subspace.max_rank,
        config.subspace.relative_singular_value_tolerance,
    )
    dependence = estimate_information_purity(
        geometry_prediction, appearance_prediction, config
    )
    nsi = float(accessibility * separation * dependence.purity)
    counts = {
        name: int(np.sum(split == name))
        for name in ("train", "validation", "test")
    }
    group_counts = {
        name: int(np.unique(dataset.groups[split == name].astype(str)).size)
        for name in ("train", "validation", "test")
    }
    return NSIResult(
        accessibility=accessibility,
        separation=separation,
        information_purity=dependence.purity,
        nsi=nsi,
        geometry_r2=geometry_r2,
        appearance_r2=appearance_r2,
        geometry_r2_per_output=geometry_r2_values,
        appearance_r2_per_output=appearance_r2_values,
        osr=osr,
        osr_metric=config.subspace.overlap_metric,
        purity_transform=config.ksg.purity_transform,
        dependence=dependence,
        geometry_alpha=geometry_alpha,
        appearance_alpha=appearance_alpha,
        geometry_rank=int(geometry_basis.shape[1]),
        appearance_raw_rank=int(appearance_raw_basis.shape[1]),
        appearance_projected_rank=int(appearance_projected_basis.shape[1]),
        split_counts=counts,
        split_group_counts=group_counts,
        geometry_basis=geometry_basis,
        appearance_raw_basis=appearance_raw_basis,
        appearance_projected_basis=appearance_projected_basis,
        geometry_probe=geometry_probe,
        appearance_probe=appearance_probe,
        validation_trace={
            "geometry": geometry_trace,
            "appearance": appearance_trace,
        },
    )


def groupwise_nsi_uncertainty(
    dataset: FeatureDataset,
    result: NSIResult,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Bootstrap held-out scene/sequence uncertainty with frozen probes."""

    dataset.validate()
    split = (
        canonical_split_array(dataset.split)
        if dataset.split is not None
        else deterministic_group_split(dataset.groups, config.split, config.seed)
    )
    test_mask = split == "test"
    test_features = dataset.features[test_mask]
    test_geometry = dataset.geometry[test_mask]
    test_appearance = dataset.appearance[test_mask]
    test_groups = dataset.groups[test_mask].astype(str)
    geometry_prediction = result.geometry_probe.predict(test_features)
    appearance_prediction = result.appearance_probe.predict(test_features)

    rows: list[dict[str, Any]] = []
    for index, group in enumerate(np.unique(test_groups)):
        mask = test_groups == group
        geometry_r2 = mean_finite(
            r2_per_output(test_geometry[mask], geometry_prediction[mask])
        )
        appearance_r2 = mean_finite(
            r2_per_output(test_appearance[mask], appearance_prediction[mask])
        )
        accessibility = float(
            (
                np.clip(geometry_r2, 0.0, 1.0)
                + np.clip(appearance_r2, 0.0, 1.0)
            )
            / 2.0
        )
        dependence = estimate_information_purity(
            geometry_prediction[mask],
            appearance_prediction[mask],
            replace(config, seed=config.seed + index + 1),
        )
        group_nsi = float(accessibility * result.separation * dependence.purity)
        rows.append(
            {
                "group": group,
                "sample_count": int(np.sum(mask)),
                "A_accessibility": accessibility,
                "B_separation_frozen": result.separation,
                "C_information_purity": dependence.purity,
                "NSI": group_nsi,
                "geometry_r2": geometry_r2,
                "appearance_r2": appearance_r2,
                "ksg_observed_mi_nats": dependence.observed_mi_nats,
                "ksg_permutation_median_mi_nats": dependence.permutation_median_mi_nats,
                "ksg_permutation_p_value": dependence.permutation_p_value,
            }
        )

    group_nsi = np.asarray([row["NSI"] for row in rows], dtype=np.float64)
    if group_nsi.size == 0:
        raise ValueError("No held-out groups are available for uncertainty estimation")
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(
        0,
        group_nsi.size,
        size=(config.statistics.bootstrap_replicates, group_nsi.size),
    )
    bootstrap = group_nsi[indices].mean(axis=1)
    tail = 100.0 * (1.0 - config.statistics.confidence_level) / 2.0
    lower, upper = np.percentile(bootstrap, [tail, 100.0 - tail])
    return {
        "unit": "held-out scene/sequence",
        "method": (
            "fixed train+validation probes; per-group held-out A and C; frozen B; "
            "equal-weight percentile group bootstrap"
        ),
        "global_NSI": result.nsi,
        "group_balanced_NSI_mean": float(group_nsi.mean()),
        "confidence_interval": [float(lower), float(upper)],
        "confidence_level": float(config.statistics.confidence_level),
        "bootstrap_replicates": int(config.statistics.bootstrap_replicates),
        "group_count": int(group_nsi.size),
        "groups": rows,
    }

