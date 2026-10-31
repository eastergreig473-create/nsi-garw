# NSI-GARW

Research code for **Weak separability of geometry and appearance features in deep image matching: diagnostics and limits of reweighting** by Liansuo Wei, Mengmeng Xu, and Weiwei Shen.

This repository contains the operational reconstruction used for the revised analyses. It is not the missing original prototype. The reconstructed NSI is a protocol-dependent diagnostic. The evaluated GARW candidate reweighting did **not** improve matching under the reported protocols. The code and results must not be read as evidence of a universal separability threshold or a guaranteed enhancement.

## Quick start

Use Python 3.10 or later in an isolated environment. From this repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e "./pipeline[test,models]"
python -m pytest pipeline/tests -q
python tools/reproduce_example.py
```

The complete test suite imports some model and image-processing dependencies; the `models` extra is therefore used above. For CPU-only recomputation of the included compact example, `python -m pip install -e ./pipeline` is sufficient. GPU extraction needs a compatible PyTorch/CUDA installation and the external assets listed in `pipeline/ASSET_REGISTRY.md`. Dependency ranges are compatibility declarations; `environment_installed_versions.json` records the actual analysis environment and is the reference for numerical comparisons.

The example checks a genuine saved HPatches SIFT held-out measurement against its metadata, verifies the input SHA-256, and recomputes A, B, C, and NSI. Its expected NSI is approximately **0.01242539896519057**. It does not require original images or model weights. `python tools/reproduce_example.py --groups` also verifies the saved group-level NSI values. These are recomputations with fixed fitted probes, not independent refits or estimates of refitting uncertainty.

## Contents

| Path | Purpose |
| --- | --- |
| `pipeline/src/nsi_garw/` | NSI, attribute extraction, matching, GARW, model adapters, and statistics |
| `pipeline/scripts/` | Experiment orchestration, compact-output recomputation, and corrected association analyses |
| `pipeline/tests/` | Scientific and data-contract regression tests |
| `pipeline/configs/` | Recorded default and full-rerun configurations |
| `pipeline/figures/` | Figure-generation source |
| `examples/` | One genuine compact held-out output with complete numerical metadata |
| `summary_data/` | Numeric figure sources and principal summary tables |
| `provenance/` | Primary reporting routes and the submitted supporting-archive index |
| `pipeline/provenance_snapshots/` | Earlier implementation snapshots, retained for audit only |
| `SHA256SUMS` | Hashes of this release's files, excluding this hash list itself |

## Reproduce the analyses

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the distinction between recomputing fixed outputs, replotting, and full feature extraction/refitting. The six S1 Data archives supplied with the manuscript contain the remaining compact measurements, predictions, labels, split manifests, and uncertainty outputs. They are journal Supporting Information; this code repository does not claim that the manuscript or those archives have already been published. Their exact names, sizes, and hashes are recorded in `provenance/submitted_supporting_archives.json` and `provenance/supporting_file_index.json`.

The frozen experiment scripts preserve historical workstation paths such as `F:/NSI_GARW_Data`. For full reruns, map these paths to your asset root and preserve the recorded group splits and asset versions. The small example above is portable and does not use those paths. Do not interpret a path embedded in provenance metadata as a downloadable public address.

## Interpretation and limitations

* NSI uses clipped held-out probe R-squared, overlap between **raw supervised** row spaces, and finite-sample bias-centered KSG dependence. The exact convention is documented in `pipeline/IMPLEMENTATION_CONTRACT.md`. PCA partitioning is not used to infer semantic geometry and appearance subspaces.
* A value of 0.5 is a descriptive operational reference, not a validated universal cutoff. Absolute values and rankings depend on the protocol and regularization.
* Candidate-reweighting results and full native matcher results have different candidate supports and must be reported separately. Poor candidate coverage can produce a floor effect.
* Static scenes and controlled corruptions do not establish robustness to real dynamic occlusions or nonrigid motion. Annotation-source agreement is not proof of equivalent ground truth.
* `provenance/PRIMARY_REPORTING_INDEX.json` identifies corrected publication outputs and superseded exploratory containers. In particular, publication association statistics use `reanalyze_associations.py` and the final complete-association outputs; the older generic bootstrap helper must not be substituted for the corrected treatment of undefined correlations.

## License and third-party assets

Author-generated code is released under the [MIT License](LICENSE). Original benchmark images, pretrained weights, external repositories, and third-party binaries are excluded. Obtain them from their recorded providers and follow their own licenses and terms. Numerical derived example/summary files are included for verification; this release does not relicense their source benchmarks. See [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md).

This repository has no assigned DOI. Use a verified commit URL to identify a specific version. `CITATION.cff` supplies software citation metadata without claiming publication of the manuscript.
