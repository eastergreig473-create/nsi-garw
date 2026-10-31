from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class RidgeConfig:
    lambdas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class SubspaceConfig:
    max_rank: int = 8
    relative_singular_value_tolerance: float = 1e-6
    overlap_metric: str = "mean_squared_principal_cosine"


@dataclass(frozen=True)
class KSGConfig:
    k: int = 5
    permutations: int = 200
    max_samples: int = 5000
    jitter: float = 1e-10
    purity_transform: str = "reciprocal_excess_nats"


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_replicates: int = 10000
    confidence_level: float = 0.95
    permutation_replicates: int = 1000


@dataclass(frozen=True)
class GARWConfig:
    penalty_lambdas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4)


@dataclass(frozen=True)
class DegradationConfig:
    strengths: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 20261023
    split: SplitConfig = field(default_factory=SplitConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    ksg: KSGConfig = field(default_factory=KSGConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    garw: GARWConfig = field(default_factory=GARWConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


def _tuple_floats(value: object, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    split = raw.get("split", {})
    ridge = raw.get("ridge", {})
    subspace = raw.get("subspace", {})
    ksg = raw.get("ksg", {})
    statistics = raw.get("statistics", {})
    garw = raw.get("garw", {})
    degradation = raw.get("degradation", {})
    config = PipelineConfig(
        seed=int(raw.get("seed", 20261023)),
        split=SplitConfig(
            train_fraction=float(split.get("train_fraction", 0.8)),
            validation_fraction=float(split.get("validation_fraction", 0.1)),
            test_fraction=float(split.get("test_fraction", 0.1)),
        ),
        ridge=RidgeConfig(
            lambdas=_tuple_floats(ridge.get("lambdas"), RidgeConfig().lambdas)
        ),
        subspace=SubspaceConfig(
            max_rank=int(subspace.get("max_rank", 8)),
            relative_singular_value_tolerance=float(
                subspace.get("relative_singular_value_tolerance", 1e-6)
            ),
            overlap_metric=str(
                subspace.get("overlap_metric", "mean_squared_principal_cosine")
            ),
        ),
        ksg=KSGConfig(
            k=int(ksg.get("k", 5)),
            permutations=int(ksg.get("permutations", 200)),
            max_samples=int(ksg.get("max_samples", 5000)),
            jitter=float(ksg.get("jitter", 1e-10)),
            purity_transform=str(
                ksg.get("purity_transform", "reciprocal_excess_nats")
            ),
        ),
        statistics=StatisticsConfig(
            bootstrap_replicates=int(statistics.get("bootstrap_replicates", 10000)),
            confidence_level=float(statistics.get("confidence_level", 0.95)),
            permutation_replicates=int(statistics.get("permutation_replicates", 1000)),
        ),
        garw=GARWConfig(
            penalty_lambdas=_tuple_floats(
                garw.get("penalty_lambdas"), GARWConfig().penalty_lambdas
            )
        ),
        degradation=DegradationConfig(
            strengths=_tuple_floats(
                degradation.get("strengths"), DegradationConfig().strengths
            )
        ),
    )
    total = (
        config.split.train_fraction
        + config.split.validation_fraction
        + config.split.test_fraction
    )
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1; received {total}")
    if not config.ridge.lambdas or any(value < 0 for value in config.ridge.lambdas):
        raise ValueError("Ridge lambdas must be a non-empty sequence of non-negative values")
    if config.ksg.k < 1:
        raise ValueError("KSG k must be positive")
    if config.subspace.max_rank < 1:
        raise ValueError("Subspace max_rank must be positive")
    if config.subspace.overlap_metric not in {
        "mean_squared_principal_cosine",
        "maximum_squared_principal_cosine",
    }:
        raise ValueError("Unsupported subspace overlap metric")
    if config.ksg.purity_transform not in {
        "reciprocal_excess_nats",
        "exp_negative_excess_nats",
    }:
        raise ValueError("Unsupported information-purity transform")
    if config.ksg.permutations < 0 or config.ksg.max_samples < config.ksg.k + 2:
        raise ValueError("Invalid KSG permutation count or sample limit")
    if config.statistics.bootstrap_replicates < 1 or config.statistics.permutation_replicates < 1:
        raise ValueError("Statistical replicate counts must be positive")
    return config

