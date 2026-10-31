"""Plot completed HPatches layer diagnostics, with sequence-level uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gen_fig_hpatches_real import ROOT, RESULTS, OUTPUT, plt, save, configure_publication_style
from nsi_garw.io import sha256_file


def main():
    configure_publication_style()
    definitions = [
        ("ResNet-50", [f"resnet50_{layer}_crop224" for layer in ("maxpool", "layer1", "layer2", "layer3", "layer4")],
         ["pool", "layer1", "layer2", "layer3", "layer4"], "ResNet-50, 224 center crop"),
        ("DINOv2-S/14", [f"dinov2_vits14_block{layer}_518" for layer in (2, 5, 8, 11)],
         ["2", "5", "8", "11"], "DINOv2-S/14, 518 input"),
    ]
    records, sources = [], []
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    for row, (model, identifiers, labels, title) in enumerate(definitions):
        group_records = []
        for identifier, label in zip(identifiers, labels):
            source = RESULTS / "layerwise" / (identifier + ".json")
            if not source.exists():
                source = RESULTS / (identifier + ".json")
            payload = json.loads(source.read_text(encoding="utf-8"))
            result, uncertainty = payload["result"], payload["groupwise_uncertainty"]
            record = {"experiment_id": identifier, "model": model, "layer": label,
                "A": result["A_accessibility"], "B": result["B_separation"],
                "C": result["C_information_purity"], "pooled_NSI": result["NSI"],
                "sequence_NSI_mean": uncertainty["group_balanced_NSI_mean"],
                "ci_low": uncertainty["confidence_interval"][0], "ci_high": uncertainty["confidence_interval"][1],
                "test_sequence_count": uncertainty["group_count"], "source_json": str(source),
                "source_sha256": sha256_file(source)}
            group_records.append(record)
            records.append(record)
            sources.append(source)
        data = pd.DataFrame(group_records)
        x = np.arange(len(labels))
        left, right = axes[row]
        for field, color, marker in (("A", "#0072B2", "o"), ("B", "#D55E00", "s"),
                                     ("C", "#009E73", "^"), ("pooled_NSI", "#000000", "D")):
            left.plot(x, data[field], color=color, marker=marker, markersize=3.5,
                      linewidth=1.3, label="NSI" if field == "pooled_NSI" else field)
        left.set_ylim(-.025, 1.05)
        left.set_ylabel("Pooled test-set value")
        left.set_title(f"{'a' if row == 0 else 'c'}  {title}", loc="left", fontsize=9)
        mean, lo, hi = (data[c].to_numpy() for c in ("sequence_NSI_mean", "ci_low", "ci_high"))
        right.errorbar(x, mean, yerr=[mean-lo, hi-mean], fmt="o-", color="#0072B2", markersize=3.5, capsize=2.5)
        right.set_ylim(-.01, .26)
        right.set_ylabel("Mean sequence NSI (95% CI)")
        right.set_title(f"{'b' if row == 0 else 'd'}  17 held-out sequences", loc="left", fontsize=9)
        for ax in (left, right):
            ax.set_xticks(x, labels)
            ax.set_xlabel("Representation stage" if row == 0 else "Transformer block (zero-based)")
            ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            left.legend(frameon=False, ncol=4, fontsize=8, loc="center right", columnspacing=.8, handlelength=1.6)
    fig.subplots_adjust(left=.085, right=.985, top=.94, bottom=.11, hspace=.62, wspace=.42)
    save(fig, "fig3_hpatches_layerwise")
    pd.DataFrame(records).to_csv(OUTPUT / "fig3_source.csv", index=False)
    provenance = {"inputs": {str(p): sha256_file(p) for p in sources},
        "script_sha256": sha256_file(Path(__file__)),
        "uncertainty": "10000 sequence-bootstrap replicates of 17 fixed-probe sequence NSI values; not uncertainty of pooled NSI or probe fitting",
        "scope": "descriptive HPatches layer diagnostics; no layer selected using test outcomes; native sampling coverage varies by stage; not MegaDepth results"}
    (OUTPUT / "fig3_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(pd.DataFrame(records)[["model", "layer", "pooled_NSI", "sequence_NSI_mean", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
