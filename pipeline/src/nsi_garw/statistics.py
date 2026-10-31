from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, wilcoxon


EPS = 1e-12


def _as_vector(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def percentile_interval(
    samples: np.ndarray | Iterable[float], confidence_level: float = 0.95
) -> tuple[float, float]:
    values = _as_vector(samples, "samples")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    tail = 100.0 * (1.0 - confidence_level) / 2.0
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    return float(lower), float(upper)


def aggregate_by_group(
    values: np.ndarray | Iterable[float], groups: np.ndarray | Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    vector = _as_vector(values, "values")
    group_array = np.asarray(groups).astype(str).reshape(-1)
    if vector.shape[0] != group_array.shape[0]:
        raise ValueError("values and groups must have equal length")
    unique = np.unique(group_array)
    means = np.asarray(
        [float(vector[group_array == group].mean()) for group in unique],
        dtype=np.float64,
    )
    return unique, means


@dataclass(frozen=True)
class PairedBootstrapResult:
    estimate: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    p_value_wilcoxon: float
    wilcoxon_statistic: float
    group_count: int
    observation_count: int
    bootstrap_replicates: int

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "estimate": self.estimate,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "p_value_wilcoxon": self.p_value_wilcoxon,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "group_count": self.group_count,
            "observation_count": self.observation_count,
            "bootstrap_replicates": self.bootstrap_replicates,
        }


def paired_group_bootstrap(
    first: np.ndarray | Iterable[float],
    second: np.ndarray | Iterable[float],
    groups: np.ndarray | Iterable[str],
    *,
    bootstrap_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Estimate mean(first-second) using groups as the resampling unit."""
    first_array = _as_vector(first, "first")
    second_array = _as_vector(second, "second")
    group_array = np.asarray(groups).astype(str).reshape(-1)
    if first_array.shape != second_array.shape or first_array.shape != group_array.shape:
        raise ValueError("first, second, and groups must have equal length")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    unique_groups, group_differences = aggregate_by_group(
        first_array - second_array, group_array
    )
    if unique_groups.size < 2:
        raise ValueError("At least two groups are required for grouped inference")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        unique_groups.size,
        size=(bootstrap_replicates, unique_groups.size),
    )
    bootstrap = group_differences[indices].mean(axis=1)
    interval = percentile_interval(bootstrap, confidence_level)
    nonzero_differences = group_differences[np.abs(group_differences) > EPS]
    if nonzero_differences.size == 0:
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            nonzero_differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return PairedBootstrapResult(
        estimate=float(group_differences.mean()),
        confidence_interval=interval,
        confidence_level=float(confidence_level),
        p_value_wilcoxon=float(p_value),
        wilcoxon_statistic=float(statistic),
        group_count=int(unique_groups.size),
        observation_count=int(first_array.size),
        bootstrap_replicates=int(bootstrap_replicates),
    )


@dataclass(frozen=True)
class AssociationResult:
    spearman_rho: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    permutation_p_value: float
    group_count: int
    bootstrap_replicates: int
    permutation_replicates: int

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "spearman_rho": self.spearman_rho,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "permutation_p_value": self.permutation_p_value,
            "group_count": self.group_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "permutation_replicates": self.permutation_replicates,
        }


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= EPS or np.std(second) <= EPS:
        return 0.0
    result = spearmanr(first, second)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def spearman_group_association(
    first: np.ndarray | Iterable[float],
    second: np.ndarray | Iterable[float],
    groups: np.ndarray | Iterable[str] | None = None,
    *,
    bootstrap_replicates: int = 10000,
    permutation_replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> AssociationResult:
    first_array = _as_vector(first, "first")
    second_array = _as_vector(second, "second")
    if first_array.shape != second_array.shape:
        raise ValueError("first and second must have equal length")
    if groups is None:
        group_array = np.arange(first_array.size).astype(str)
    else:
        group_array = np.asarray(groups).astype(str).reshape(-1)
        if group_array.shape != first_array.shape:
            raise ValueError("groups must be aligned with the observations")
    unique_groups = np.unique(group_array)
    grouped_first = np.asarray(
        [first_array[group_array == group].mean() for group in unique_groups]
    )
    grouped_second = np.asarray(
        [second_array[group_array == group].mean() for group in unique_groups]
    )
    if unique_groups.size < 4:
        raise ValueError("At least four independent groups are required for association")
    observed = _safe_spearman(grouped_first, grouped_second)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        selected = rng.integers(0, unique_groups.size, unique_groups.size)
        bootstrap[index] = _safe_spearman(
            grouped_first[selected], grouped_second[selected]
        )
    null = np.empty(permutation_replicates, dtype=np.float64)
    for index in range(permutation_replicates):
        null[index] = _safe_spearman(
            grouped_first, grouped_second[rng.permutation(unique_groups.size)]
        )
    p_value = float(
        (1 + np.sum(np.abs(null) >= abs(observed))) / (null.size + 1)
    )
    return AssociationResult(
        spearman_rho=observed,
        confidence_interval=percentile_interval(bootstrap, confidence_level),
        confidence_level=float(confidence_level),
        permutation_p_value=p_value,
        group_count=int(unique_groups.size),
        bootstrap_replicates=int(bootstrap_replicates),
        permutation_replicates=int(permutation_replicates),
    )


@dataclass(frozen=True)
class RankStabilityResult:
    variant: str
    spearman_rho: float
    kendall_tau: float
    item_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "variant": self.variant,
            "spearman_rho": self.spearman_rho,
            "kendall_tau": self.kendall_tau,
            "item_count": self.item_count,
        }


def ranking_stability(
    baseline_scores: np.ndarray | Iterable[float],
    variant_scores: dict[str, np.ndarray | Iterable[float]],
) -> list[RankStabilityResult]:
    baseline = _as_vector(baseline_scores, "baseline_scores")
    baseline_ranks = rankdata(-baseline, method="average")
    results: list[RankStabilityResult] = []
    for name, raw in variant_scores.items():
        values = _as_vector(raw, name)
        if values.shape != baseline.shape:
            raise ValueError(f"Variant {name!r} is not aligned with the baseline")
        ranks = rankdata(-values, method="average")
        spearman = spearmanr(baseline_ranks, ranks).statistic
        kendall = kendalltau(baseline_ranks, ranks).statistic
        results.append(
            RankStabilityResult(
                variant=str(name),
                spearman_rho=float(spearman) if np.isfinite(spearman) else 0.0,
                kendall_tau=float(kendall) if np.isfinite(kendall) else 0.0,
                item_count=int(baseline.size),
            )
        )
    return results


def benjamini_hochberg(p_values: np.ndarray | Iterable[float]) -> np.ndarray:
    values = _as_vector(p_values, "p_values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted
