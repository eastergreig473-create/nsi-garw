"""Fig 5: three within-representation associations from real held-out sequences."""
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsi_garw.figures import configure_publication_style
from nsi_garw.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/real"
OUTPUT = ROOT / "figures/real"


def main():
    configure_publication_style()
    source = RESULTS / "native_matchers/native_analysis_complete.json"
    native = json.loads(source.read_text(encoding="utf-8"))
    if native["status"] != "PASS" or len(native["associations"]) != 2:
        raise ValueError("Native association analysis is incomplete")
    sift_path = RESULTS / "association/sift_clean_association.json"
    sift = json.loads(sift_path.read_text(encoding="utf-8"))["analysis"]
    sources = [("SIFT", RESULTS / "association/sift_clean_sequence_association.csv", sift)]
    sources += [("LoFTR" if model == "loftr" else "RoMa",
                 RESULTS / f"native_matchers/{model}_nsi_native_MMA3_points.csv",
                 next(row for row in native["associations"] if row["model"] == model))
                for model in ("loftr", "roma")]
    audited=RESULTS/"association_audit/analysis_complete.json"
    if audited.exists():
        review=json.loads(audited.read_text(encoding="utf-8"))
        revised=[]
        for name,path,stats in sources:
            row=next(r for r in review["rows"] if r["dataset"]=="HPatches" and r["representation"]==name)
            if row["confidence_interval"] is None:
                raise ValueError("HPatches association interval is not eligible for Fig5")
            row=dict(row)
            if name!="SIFT":row["BH_q_2_native_associations"]=row["BH_q_two_native_associations"]
            revised.append((name,path,row))
        sources=revised
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.6), sharey=True)
    point_frames, sequence_names = [], None
    for index, (ax, (name, path, stats)) in enumerate(zip(axes, sources)):
        points = pd.read_csv(path)
        if len(points) != 17 or points.scene.nunique() != 17:
            raise ValueError("Fig5 needs all 17 unique held-out sequences")
        names = set(points.scene)
        if sequence_names is not None and names != sequence_names:
            raise ValueError("Association panels do not use the same sequences")
        sequence_names = names
        for stratum, marker, color, prefix in [("Illumination", "^", "#0072B2", "i_"),
                                               ("Viewpoint", "o", "#D55E00", "v_")]:
            block = points[points.scene.str.startswith(prefix)]
            ax.scatter(block.NSI, block.mma_at_3*100, label=stratum, marker=marker,
                       color=color, s=23, alpha=.9, edgecolors="white", linewidths=.25)
        ci = stats["confidence_interval"]
        subtitle = f"ρ = {stats['spearman_rho']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
        probability = f"p = {stats['permutation_p_value']:.3f}"
        if index:
            probability += f"; q = {stats['BH_q_2_native_associations']:.3f}"
        ax.set_title(f"{chr(97+index)}  {name}\n{subtitle}\n{probability}", loc="left", fontsize=8.4, pad=8)
        ax.set_xlabel("Sequence NSI", fontsize=8.5)
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(alpha=.14)
        point_frames.append(points.assign(panel=chr(97+index), representation=name))
    axes[0].set_ylabel("MMA at 3 pixels (%)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.2,
               bbox_to_anchor=(.53, -.005))
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.73, wspace=.20)
    for extension in ("png", "pdf", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(OUTPUT / ("fig5_sift_association."+extension), dpi=600, **options)
    plt.close(fig)
    pd.concat(point_frames, ignore_index=True).to_csv(OUTPUT / "fig5_source.csv", index=False)
    inputs = [source, sift_path, *[path for _, path, _ in sources]]
    if audited.exists():inputs.append(audited)
    (OUTPUT / "fig5_provenance.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256_file(Path(__file__)),
        "inputs": {str(path): sha256_file(path) for path in inputs}, "point_count": 51,
        "uncertainty": "95% sequence bootstrap CI for Spearman rho;10000 replicates;two-sided1000-permutation p",
        "multiple_testing": "BH q applies only to two native matcher associations; SIFT was analyzed earlier separately; exploratory families declared after outcomes",
        "scope": "Within-representation associations on identical17 sequences, not cross-model prediction; SIFT MNN and complete native matchers have different candidate/output protocols"}, indent=2), encoding="utf-8")
    print("Fig5: 51 real sequence points in three panels; PNG/PDF/TIFF/source/provenance saved", flush=True)


if __name__ == "__main__":
    main()
