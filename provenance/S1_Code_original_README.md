# Supporting code and source data

Manuscript: PONE-D-26-43913. The files document an auditable reconstruction,
including negative results. No synthetic test fixture is manuscript evidence.

S1 Code contains the Python source, figure generators, tests, frozen TOML
configuration, asset/source registries, and installed package versions. S1 Data
and its numbered ZIP parts contain result JSON, per-pair/per-scene measurements,
candidate scores and labels, held-out NSI measurements, split and preprocessing
audits, provenance hashes, and numeric figure source tables. Each ZIP is a
complete standard archive smaller than 20 MB; extract all data parts to the same
directory. The external file_index.json maps every file to an archive and SHA-256.

The package is local Supporting Information prepared for journal upload. It does
not claim that a repository has already been published or that a DOI exists.

## Reproduction levels

1. Replot and inspect reported results using `measurements/` and `figure_source/`.
   The primary experimental unit is a held-out sequence (HPatches) or scene
   (ETH3D/MegaDepth). Do not substitute image features for independent groups.
2. Recompute NSI from `measurements/supporting_measurements/`. Each NPZ retains
   every held-out attribute target and prediction, scene and sample identifiers,
   fitted raw geometry/appearance bases, coefficients, and training normalization.
   The adjacent JSON supplies the exact configuration, source-result hash,
   validation traces, and selected regularization. Compute A from the two held-out
   R-squared values, B from raw supervised principal-cosine overlap, and C from
   the provided predictions using the seeded KSG implementation. Per-scene C
   uses master seed plus sorted-group index plus one. Bootstrap group values
   with fixed probes, as stated in the manuscript. Reconstructing fitted A/B was
   checked against the original saved results before each compact file was made.
   Run `python scripts/recompute_compact_nsi.py <measurement.npz> --groups`
   from the installed pipeline directory to verify global and group estimates.
3. Refit probes or rerun feature extraction from public assets. Install the
   recorded Python 3.10/CUDA-compatible package versions, then install the source
   with `python -m pip install -e pipeline --no-deps`. Obtain original datasets,
   weights, and external repositories from the source registries and verify all
   pinned hashes/commits. No training or fine-tuning of the pretrained models is
   part of this reconstruction. Run scripts from the pipeline directory.

Bulk data originally lived at `F:/NSI_GARW_Data`; this path is explicit in
`scripts/run_hpatches_suite.py` and several preparation scripts. Recreate that
layout on Windows or consistently replace the bulk-root path and manifests in a
new reproduction workspace. Do not change paths or code in a running frozen
experiment. Native full-matchers, encoder candidate reranking, and NSI attribute
probes have distinct inputs and inference boundaries.

Original datasets: HPatches <https://github.com/hpatches/hpatches-dataset>, ETH3D
<https://www.eth3d.net/datasets>, MegaDepth
<https://www.cs.cornell.edu/projects/megadepth/>. Images, pretrained weights,
external Git checkouts, portable COLMAP binaries, and full native descriptor
caches are not redistributed. The original third-party sources and the provided
code regenerate those assets and caches. Feature export hashes identify exact
inputs; compact test measurements permit auditing the reported NSI without
redistributing the large descriptor caches.

## Main execution chain

The task-specific scripts retain their frozen protocols, checkpoint checks, split
manifests, and SHA-256 input/result dependencies. HPatches uses the 81/18/17
sequence partition. ETH3D uses the 5/3/5 public-scene partition. MegaDepth scene
labels are frozen before depth eligibility checks, without replacement or
reshuffling. See each saved protocol for outcome timing and exploratory status.

- `run_hpatches_suite.py`, `run_sift_degradation_matching.py`,
  `run_sift_degradation_nsi.py`, and `run_deep_texture_nsi.py`: feature/NSI and
  corruption measurements.
- `verify_completed_nsi.py` and `verify_completed_degradations.py`: stable-core
  verification of early HPatches results. Their saved verification records
  supersede early source-version metadata where point values agree.
- `run_eth3d_colmap_dense.py`, then `run_eth3d_dense_sensitivity.py`: fixed-pose
  dense reconstruction and common-grid laser/COLMAP annotation sensitivity.
- `prepare_megadepth_cohort.py`, then `run_megadepth_revision.py` in its CPU,
  GPU, and analysis modes: outcome-blind cohort, PCA, ten models and stages.
- Native paired matching and auxiliary encoder matching have separate result
  directories; `analyze_complete_hpatches_associations.py` audits all fifteen
  declared HPatches associations. `reanalyze_associations.py` corrects undefined
  correlation handling without changing the NSI core. ETH3D five-scene p values
  enumerate all 120 label permutations; untestable vectors remain undefined.
- `benchmark_online_garw.py`: matched online encoder-pipeline runtime, FP32 and
  AMP FP16. Native matching latency is reported with its separate boundary.

The generic core's legacy correlation helper assigns zero to a constant-vector
correlation. Publication association analyses explicitly use the independent
audited helper instead: undefined draws are never treated as measured zeros.
Do not use the legacy generic helper to recreate publication correlation CIs.

The package preserves exploratory/superseded outputs for audit. Use the completed
analysis markers, `pipeline_manifests/PRIMARY_REPORTING_INDEX.json`, and manuscript
source tables to identify primary reporting. The index separates corrected
publication associations from older PASS-marked analysis containers;
mixed-precision prototypes and failed sparse-annotation attempts do not replace
the specified primary experiments.
