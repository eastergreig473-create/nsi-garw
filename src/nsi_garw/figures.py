from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6B7280",
}


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )


def _save_all(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf", ".tiff"):
        destination = stem.with_suffix(suffix)
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == ".tiff" else {}
        fig.savefig(destination, **options)
        paths.append(str(destination.resolve()))
    plt.close(fig)
    return paths


def _collect_csv(root: Path, filename: str) -> pd.DataFrame | None:
    paths = sorted(root.rglob(filename))
    if not paths:
        return None
    blocks = []
    for path in paths:
        table = pd.read_csv(path)
        table["source_file"] = str(path.resolve())
        blocks.append(table)
    return pd.concat(blocks, ignore_index=True)


def figure_nsi_components(table: pd.DataFrame, output: Path) -> list[str]:
    required = {"model", "A_accessibility", "B_separation", "C_information_purity", "NSI"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"NSI component figure is missing columns: {missing}")
    grouped = (
        table.groupby("model", as_index=False)[
            ["A_accessibility", "B_separation", "C_information_purity", "NSI"]
        ]
        .mean()
        .sort_values("NSI", ascending=False)
    )
    labels = grouped["model"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.19
    fig, axis = plt.subplots(figsize=(max(6.75, 0.62 * len(labels)), 3.65))
    components = [
        ("A_accessibility", "A: accessibility", PALETTE["blue"]),
        ("B_separation", "B: separation", PALETTE["orange"]),
        ("C_information_purity", "C: information purity", PALETTE["green"]),
        ("NSI", "NSI", PALETTE["red"]),
    ]
    hatches = ["", "//", "..", "xx"]
    for index, (column, label, color) in enumerate(components):
        bars = axis.bar(
            x + (index - 1.5) * width,
            grouped[column],
            width,
            label=label,
            color=color,
            hatch=hatches[index],
            edgecolor="white",
            linewidth=0.45,
        )
        if column == "NSI" and len(labels) <= 12:
            axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, color="#444444")
    axis.axhline(0.5, color=PALETTE["gray"], linewidth=0.8, linestyle="--", label="0.5 reference")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.04)
    wrapped = [textwrap.fill(label.replace("_", " "), width=15) for label in labels]
    axis.set_xticks(x, wrapped, rotation=24, ha="right")
    axis.set_title("Cross-model NSI decomposition", loc="left", pad=10)
    axis.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return _save_all(fig, output / "figure_2_nsi_components")


def figure_depth_curve(table: pd.DataFrame, output: Path) -> list[str]:
    depth_column = "depth" if "depth" in table.columns else "layer_index"
    required = {"model", depth_column, "NSI"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Depth figure is missing columns: {missing}")
    fig, axis = plt.subplots(figsize=(6.2, 3.6))
    colors = list(PALETTE.values())
    for index, (model, block) in enumerate(table.groupby("model", sort=True)):
        grouped = block.groupby(depth_column, as_index=False)["NSI"].mean().sort_values(depth_column)
        axis.plot(
            grouped[depth_column],
            grouped["NSI"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=str(model),
            color=colors[index % len(colors)],
        )
    axis.set_xlabel("Normalized depth" if depth_column == "depth" else "Layer index")
    axis.set_ylabel("NSI")
    axis.set_ylim(0.0, 1.04)
    axis.set_title("NSI across representation depth")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output / "figure_3_depth_curve")


def figure_degradation_curves(table: pd.DataFrame, output: Path) -> list[str]:
    required = {"condition", "strength", "NSI"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Degradation figure is missing columns: {missing}")
    if table["strength"].nunique() < 2:
        raise ValueError("Degradation figure requires at least two strength levels")
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    colors = list(PALETTE.values())
    for index, (condition, block) in enumerate(table.groupby("condition", sort=True)):
        grouped = block.groupby("strength")["NSI"].agg(["mean", "sem"]).reset_index()
        color = colors[index % len(colors)]
        axis.plot(grouped["strength"], grouped["mean"], marker="o", linewidth=1.4, label=str(condition), color=color)
        uncertainty = grouped["sem"].fillna(0.0).to_numpy()
        axis.fill_between(
            grouped["strength"],
            grouped["mean"] - uncertainty,
            grouped["mean"] + uncertainty,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    axis.set_xlabel("Degradation strength")
    axis.set_ylabel("NSI")
    axis.set_ylim(0.0, 1.04)
    axis.set_title("Representation robustness under controlled degradation")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output / "figure_4_degradation_curves")


def figure_nsi_performance(table: pd.DataFrame, output: Path) -> list[str]:
    performance_candidates = [
        "performance",
        "MMA",
        "mma",
        "AUC",
        "auc",
        "top1",
        "average_precision",
    ]
    performance_column = next(
        (column for column in performance_candidates if column in table.columns), None
    )
    if "NSI" not in table.columns or performance_column is None:
        raise ValueError("Association figure requires NSI and a performance column")
    x = table["NSI"].to_numpy(dtype=np.float64)
    y = table[performance_column].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig, axis = plt.subplots(figsize=(4.6, 3.8))
    axis.scatter(x, y, color=PALETTE["blue"], s=24, alpha=0.75, edgecolor="white", linewidth=0.4)
    if x.size >= 2 and np.std(x) > 0.0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        axis.plot(grid, slope * grid + intercept, color=PALETTE["red"], linewidth=1.5)
    axis.set_xlabel("NSI")
    axis.set_ylabel(str(performance_column))
    axis.set_title("NSI–performance association")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output / "figure_5_nsi_performance")


def figure_garw(per_scene: pd.DataFrame, output: Path) -> list[str]:
    required = {"scene", "method", "average_precision"}
    missing = sorted(required.difference(per_scene.columns))
    if missing:
        raise ValueError(f"GARW figure is missing columns: {missing}")
    preferred = ["geometry", "appearance", "garw_lambda_zero", "garw", "oracle_branch"]
    methods = [method for method in preferred if method in set(per_scene["method"])]
    if not methods:
        methods = sorted(per_scene["method"].astype(str).unique())
    grouped = per_scene.groupby("method")["average_precision"].agg(["mean", "sem"])
    means = np.asarray([grouped.loc[method, "mean"] for method in methods], dtype=float)
    rng_ci = np.random.default_rng(20261023)
    intervals = []
    for method in methods:
        values = per_scene.loc[per_scene["method"] == method, "average_precision"].to_numpy(dtype=float)
        selected = rng_ci.integers(0, values.size, size=(10000, values.size))
        bootstrap = values[selected].mean(axis=1)
        intervals.append(np.percentile(bootstrap, [2.5, 97.5]))
    intervals_array = np.asarray(intervals)
    errors = np.vstack([means - intervals_array[:, 0], intervals_array[:, 1] - means])
    method_colors = {
        "geometry": PALETTE["gray"],
        "appearance": PALETTE["orange"],
        "garw_lambda_zero": PALETTE["sky"],
        "garw": PALETTE["red"],
        "oracle_branch": PALETTE["green"],
    }
    colors = [method_colors.get(method, PALETTE["blue"]) for method in methods]
    fig, axis = plt.subplots(figsize=(5.8, 3.7))
    x = np.arange(len(methods))
    axis.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="white", linewidth=0.5)
    rng = np.random.default_rng(0)
    for index, method in enumerate(methods):
        values = per_scene.loc[per_scene["method"] == method, "average_precision"].to_numpy(dtype=float)
        jitter = rng.normal(scale=0.035, size=values.size)
        axis.scatter(np.full(values.size, index) + jitter, values, s=10, color="black", alpha=0.35, linewidth=0)
    display_names = {
        "geometry": "Geometry",
        "appearance": "Appearance",
        "garw_lambda_zero": "GARW\n($\\lambda=0$)",
        "garw": "GARW",
        "oracle_branch": "Oracle\nbranch",
    }
    axis.set_xticks(x, [display_names.get(method, method.replace("_", "\n")) for method in methods])
    axis.set_ylabel("Scene-level average precision")
    axis.set_ylim(0.0, 1.04)
    axis.set_title("GARW evaluation and ablations", loc="left", pad=10)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output / "figure_6_garw")


def figure_matching_mma(per_scene: pd.DataFrame, output: Path) -> list[str]:
    required = {"scene", "method"}
    missing = sorted(required.difference(per_scene.columns))
    if missing:
        raise ValueError(f"Matching figure is missing columns: {missing}")
    metric_columns = [column for column in per_scene.columns if column.startswith("mma_at_")]
    if not metric_columns:
        raise ValueError("Matching figure contains no mma_at_* columns")
    thresholds = np.asarray([float(column.removeprefix("mma_at_")) for column in metric_columns])
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    metric_columns = [metric_columns[index] for index in order]
    fig, axis = plt.subplots(figsize=(5.6, 3.8))
    colors = list(PALETTE.values())
    for index, (method, block) in enumerate(per_scene.groupby("method", sort=True)):
        means = block[metric_columns].mean(axis=0).to_numpy(dtype=float)
        sem = block[metric_columns].sem(axis=0).fillna(0.0).to_numpy(dtype=float)
        color = colors[index % len(colors)]
        axis.plot(thresholds, means, marker="o", linewidth=1.5, label=str(method), color=color)
        axis.fill_between(thresholds, means - sem, means + sem, color=color, alpha=0.15, linewidth=0)
    axis.set_xlabel("Reprojection threshold (pixels)")
    axis.set_ylabel("Mean matching accuracy")
    axis.set_ylim(0.0, 1.04)
    axis.set_title("Held-out matching accuracy by reprojection threshold")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output / "figure_6_matching_mma")


def generate_figures(results_directory: str | Path, output_directory: str | Path) -> dict[str, object]:
    configure_publication_style()
    root = Path(results_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"figures": {}, "skipped": {}, "sources": {}}
    nsi = _collect_csv(root, "nsi_results.csv")
    garw_scene = _collect_csv(root, "garw_per_scene.csv")
    matching_scene = _collect_csv(root, "matching_per_scene.csv")
    tasks: list[tuple[str, pd.DataFrame | None, Callable[[pd.DataFrame, Path], list[str]]]] = [
        ("figure_2_nsi_components", nsi, figure_nsi_components),
        ("figure_3_depth_curve", nsi, figure_depth_curve),
        ("figure_4_degradation_curves", nsi, figure_degradation_curves),
        ("figure_5_nsi_performance", nsi, figure_nsi_performance),
        ("figure_6_garw", garw_scene, figure_garw),
        ("figure_6_matching_mma", matching_scene, figure_matching_mma),
    ]
    for name, table, function in tasks:
        if table is None:
            manifest["skipped"][name] = "source CSV not found"  # type: ignore[index]
            continue
        try:
            manifest["figures"][name] = function(table, output)  # type: ignore[index]
            manifest["sources"][name] = sorted(table["source_file"].unique().tolist())  # type: ignore[index]
        except ValueError as error:
            manifest["skipped"][name] = str(error)  # type: ignore[index]
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
