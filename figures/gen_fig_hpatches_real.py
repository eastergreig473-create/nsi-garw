"""Publication figures from completed real runs; no simulated or legacy values."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsi_garw.figures import configure_publication_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/real"
OUTPUT = ROOT / "figures/real"
SHORT = {"sift_default": "SIFT", "xfeat_e92685f_top4096_thr005": "XFeat",
    "superpoint_eb42fee_top4096_resize1024": "SuperPoint", "dinov2_vits14_block11_518": "DINOv2-S/14",
    "dedode_l_v2_b_top4096_resize784": "DeDoDe", "loftr_outdoor_coarse_1024": "LoFTR encoder",
    "roma_outdoor_scale16_560": "RoMa encoder", "vggt_1b_layer23_518": "VGGT tokens",
    "mast3r_encoder23_512": "MASt3R encoder", "dust3r_encoder23_512": "DUSt3R encoder"}


def save(fig, name):
    for extension in ("pdf", "png", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(OUTPUT / (name + "." + extension), dpi=600, **options)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    source = RESULTS / "hpatches_nsi_summary.csv"
    data = pd.read_csv(source)
    eth_source = RESULTS / "eth3d_nsi_summary.csv"
    cohorts = [("HPatches", data, 17)]
    if eth_source.exists():
        cohorts.append(("ETH3D", pd.read_csv(eth_source), 5))
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(7.0, 3.4*len(cohorts)),
        squeeze=False, gridspec_kw={"width_ratios": [1.3, 1]})
    columns = ["A_accessibility", "B_separation", "C_information_purity", "NSI"]
    for row, (dataset, cohort, groups) in enumerate(cohorts):
        left, right = axes[row]
        labels = [SHORT[s] for s in cohort.experiment_id]
        values = cohort[columns].to_numpy()
        display = left.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        left.set_yticks(np.arange(len(cohort)), labels, fontsize=7.5)
        left.set_xticks(range(4), ["A", "B", "C", "NSI"])
        left.set_title(f"{'a' if row == 0 else 'c'}  {dataset}: pooled components", loc="left", fontsize=9)
        left.grid(False)
        for i in range(len(cohort)):
            for j in range(4):
                number = f"{values[i, j]:.4f}" if row > 0 and j == 3 else f"{values[i, j]:.3f}"
                left.text(j, i, number, va="center", ha="center", fontsize=7,
                          color="white" if values[i, j] > .6 else "#182b3a")
        mean = cohort.group_NSI_mean.to_numpy()
        right.errorbar(mean, np.arange(len(cohort)), xerr=np.vstack([mean-cohort.group_NSI_CI_low, cohort.group_NSI_CI_high-mean]),
            fmt="o", color="#0072B2", capsize=2.5, markersize=3.5)
        right.set_ylim(len(cohort)-.5, -.5)
        right.set_yticks([])
        right.set_xlim(0, .22 if row == 0 else .10)
        right.set_xlabel("Mean sequence NSI (95% CI)" if row == 0 else "Mean scene NSI (95% CI)", fontsize=8)
        right.set_title(f"{'b' if row == 0 else 'd'}  {groups} held-out {'sequences' if row == 0 else 'scenes'}", loc="left", fontsize=9)
        right.grid(axis="y", visible=False)
        right.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=.19, right=.98, bottom=.13, top=.94, wspace=.20, hspace=.36)
    cax = fig.add_axes([.25, .055, .55, .016])
    fig.colorbar(display, cax=cax, orientation="horizontal", label="Component value")
    save(fig, "fig2_hpatches_nsi")
    pd.concat([cohort.assign(dataset=dataset) for dataset, cohort, _ in cohorts]).to_csv(OUTPUT / "fig2_source.csv", index=False)

    stress_rows = []
    for strength in (0., .2, .4, .6, .8, 1.):
        identifier = f"sift_degraded_texture_s{round(strength * 100):03d}"
        nsi_path = RESULTS / ("sift_default.json" if strength == 0 else f"sift_degradation_nsi/{identifier}.json")
        matching_path = RESULTS / ("descriptor_matching/sift_default/matching_summary.json" if strength == 0 else f"sift_degradation_matching/{identifier}/matching_summary.json")
        if not nsi_path.exists() or not matching_path.exists():
            continue
        nsi = json.loads(nsi_path.read_text(encoding="utf-8"))
        matching = json.loads(matching_path.read_text(encoding="utf-8"))["results"]["held_out_test_sequences"]["metrics"]["mma_at_3"]
        uncertainty = nsi["groupwise_uncertainty"]
        coverage = 580
        if strength:
            coverage = json.loads(nsi_path.with_suffix(".build.json").read_text(encoding="utf-8"))["pairs_with_aligned_features"]
        stress_rows.append({"strength": strength, "global_NSI": nsi["result"]["NSI"],
            "sequence_NSI_mean": uncertainty["group_balanced_NSI_mean"],
            "NSI_ci_low": uncertainty["confidence_interval"][0], "NSI_ci_high": uncertainty["confidence_interval"][1],
            "MMA_at_3": matching["estimate"], "MMA_ci_low": matching["confidence_interval"][0],
            "MMA_ci_high": matching["confidence_interval"][1], "pairs_with_aligned_features": coverage,
            "nsi_json": str(nsi_path), "matching_json": str(matching_path)})
    if len(stress_rows) == 6:
        stress = pd.DataFrame(stress_rows)
        fig, axes = plt.subplots(1, 2, figsize=(7., 3.0), sharex=True)
        for ax, mean_column, lo_column, hi_column, label, title, multiplier in [
            (axes[0], "sequence_NSI_mean", "NSI_ci_low", "NSI_ci_high", "Mean sequence NSI", "a  Separability of surviving features", 1),
            (axes[1], "MMA_at_3", "MMA_ci_low", "MMA_ci_high", "MMA at 3 pixels (%)", "b  Matching with all pairs", 100)]:
            mean = stress[mean_column].to_numpy() * multiplier
            lo, hi = stress[lo_column].to_numpy() * multiplier, stress[hi_column].to_numpy() * multiplier
            ax.plot(stress.strength, mean, "o-", color="#0072B2", markersize=4)
            ax.fill_between(stress.strength, lo, hi, color="#0072B2", alpha=.16)
            ax.set_xlabel("Texture-suppression strength")
            ax.set_ylabel(label)
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_xticks([0, .2, .4, .6, .8, 1])
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, "fig4_sift_texture")
        stress.to_csv(OUTPUT / "fig4_source.csv", index=False)

    association_path = RESULTS / "association/sift_clean_association.json"
    association_csv = RESULTS / "association/sift_clean_sequence_association.csv"
    if association_path.exists():
        association = json.loads(association_path.read_text(encoding="utf-8"))["analysis"]
        points = pd.read_csv(association_csv)
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for condition, marker, color, prefix in [("Illumination", "^", "#0072B2", "i_"), ("Viewpoint", "o", "#D55E00", "v_")]:
            block = points[points.scene.str.startswith(prefix)]
            ax.scatter(block.NSI, block.mma_at_3 * 100, label=condition, color=color, marker=marker, s=30)
        ax.set_xlabel("Sequence NSI")
        ax.set_ylabel("Sequence MMA at 3 pixels (%)")
        ax.set_title("Clean SIFT: 17 held-out sequences", loc="left")
        ax.text(.98, .98, f"Spearman ρ = {association['spearman_rho']:.3f}\nPermutation p = {association['permutation_p_value']:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, "fig5_sift_association")
        points.to_csv(OUTPUT / "fig5_source.csv", index=False)

    folder = RESULTS / "garw_hpatches/loftr_outdoor_coarse_1024/non_oracle"
    per_scene = pd.read_csv(folder / "garw_per_scene.csv")
    methods = ["descriptor_baseline", "geometry", "appearance", "fixed_weight", "garw_lambda_zero", "garw"]
    method_labels = ["Raw descriptor", "Geometry only", "Appearance only", "Fixed gate", "Gate, λ = 0", "Full GARW"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.35), gridspec_kw={"width_ratios": [1.35, 1]})
    source_rows = []
    rng = np.random.default_rng(20261023)
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        block = per_scene[per_scene.method == method].sort_values("scene")
        values = block.top1.to_numpy() * 100
        boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        mean = values.mean()
        color = "#D55E00" if method == "garw" else "#0072B2"
        left.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color, capsize=3, markersize=4)
        source_rows.append({"method": method, "mean_top1_percent": mean, "ci_low": lo, "ci_high": hi})
    left.set_yticks(range(len(methods)), method_labels)
    left.invert_yaxis()
    left.set_xlabel("Top-1 correct (%)")
    left.set_title("a  Non-oracle candidate reranking", loc="left")
    left.grid(axis="y", visible=False)
    pivot = per_scene.pivot(index="scene", columns="method", values="top1").sort_index()
    difference = 100 * (pivot.garw - pivot.descriptor_baseline)
    right.axhline(0, linestyle="--", color="#6B7280", linewidth=.8)
    right.scatter(np.arange(1, len(difference)+1), difference, s=23, color="#D55E00", marker="s")
    right.set_xticks([1, 5, 9, 13, 17])
    right.set_xlabel("Held-out sequence index")
    right.set_ylabel("GARW − raw descriptor (pp)")
    right.set_title("b  Paired sequence differences", loc="left")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.19, right=.98, bottom=.20, top=.87, wspace=.50)
    save(fig, "fig6_hpatches_garw")
    pd.DataFrame(source_rows).to_csv(OUTPUT / "fig6_source.csv", index=False)
    pivot.assign(difference_pp=difference).to_csv(OUTPUT / "fig6_per_sequence.csv")
    inputs = [source, folder / "garw_per_scene.csv", folder / "garw_summary.json"]
    if eth_source.exists():
        inputs.append(eth_source)
    inputs += [Path(row[key]) for row in stress_rows for key in ("nsi_json", "matching_json")]
    if association_path.exists():
        inputs += [association_path, association_csv]
    (OUTPUT / "figure_provenance.json").write_text(json.dumps({
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fig2_uncertainty": "10000 bootstraps of 17 HPatches sequences or five ETH3D scenes with fixed probes; not uncertainty of pooled NSI or probe fitting; datasets and attribute constructions differ",
        "fig6_uncertainty": "sequence bootstrap, 10000 replicates; shared fixed candidate sets; no positive-candidate injection",
        "scope": "completed HPatches runs only"}, indent=2), encoding="utf-8")
    print(f"Generated real figures for {len(data)} completed representations")
    if (RESULTS / "native_matchers/native_analysis_complete.json").exists():
        # Preserve the complete three-panel association figure on routine reruns.
        from gen_fig_native_associations import main as native_association_figure
        native_association_figure()


if __name__ == "__main__":
    main()
