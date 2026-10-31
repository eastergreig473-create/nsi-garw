# Reproducibility guide

## 1. Verify this source package

Run `python tools/verify_release.py` from the repository root. It checks every SHA-256 entry in `SHA256SUMS`. The aggregate package-source fingerprint used for the analyses is `33cca14b7e8b98f26c9d1f81e89da2b2f8606a0d1ea9df37326d05053d8d62aa`. As implemented by `source_sha256()` in `experiments.py`, this hashes each sorted `pipeline/src/nsi_garw/*.py` filename followed by its file bytes. It is not the hash of one file. Source snapshots are not the current implementation.

## 2. Recompute one genuine result without the original assets

Install the core dependencies, then run `python tools/reproduce_example.py`. Add `--groups` to check all saved sequence-level results as well. The NPZ/JSON pair was copied byte-for-byte from the completed compact-measurement export, not generated as synthetic demonstration data. The metadata preserves the configuration, source hashes, target and probe conventions, and original result path.

This level of reproduction checks NSI calculated from fixed targets, fixed predictions, and fixed supervised bases. It does not regenerate the image features, annotation targets, fitted ridge probes, or full feature caches. Its uncertainty is conditional on the saved fits. Expect machine-precision differences across library versions; the checker uses the original recorded 1e-8 absolute/relative tolerance.

## 3. Recompute the remaining compact measurements

Obtain all six S1 Data archives from the manuscript's Supporting Information when available. Extract all of them into one directory, preserving internal paths. Use `provenance/supporting_file_index.json` to locate each paired compact NPZ/JSON entry; the archive column identifies the required part. The executable script is:

```sh
python pipeline/scripts/recompute_compact_nsi.py PATH_TO_MEASUREMENT.npz --groups
```

Install the package first so that `nsi_garw` resolves. The included file index is the exact historical supporting-package index; it is not a claim that all its entries live in this Git checkout. `submitted_supporting_archives.json` describes the frozen original supporting package before the public-code release was assembled.

## 4. Replot and reanalyse

Numeric figure CSVs are in `summary_data/figure_sources/`. Full figure generators also load the completed analysis JSONs in the S1 Data archives. Restore those under `pipeline/results/real/`, following the file index. The original journal archive prefix `pipeline/` maps directly to this repository's `pipeline/` directory. Run the baseline real-source generator before the revision extension generator; extensions must run last. Consult `provenance/PRIMARY_REPORTING_INDEX.json` for the final result routes and superseded outputs.

The archived implementation contract describes general statistics. For final correlations, the primary reporting index and the corrected audit scripts take precedence: undefined constant-vector correlations are not imputed as measured zeros, and the five-scene ETH3D analysis does not report an ordinary bootstrap correlation interval. Multiple-comparison families must be kept intact.

## 5. Refit from the public datasets

Read `pipeline/ASSET_REGISTRY.md`, `pipeline/MODEL_ADAPTERS.md`, `pipeline/IMPLEMENTATION_CONTRACT.md`, and `pipeline/README.md`. Acquire HPatches, ETH3D, MegaDepth, the recorded model checkpoints, and any external matcher repositories from those sources. Preserve their version and weight hashes. The original assets and native feature caches are large and are not included in this code release.

The frozen orchestration scripts contain historical absolute paths; change the asset-root constants to the local locations before executing them. Inspect configurations rather than launching the complete suite blindly. The main entry points include `run_hpatches_suite.py`, `run_eth3d_suite.py`, `run_megadepth_revision.py`, `run_hpatches_garw.py`, `run_eth3d_garw_and_associations.py`, and the subsequent audit scripts identified in the primary index. This is a documented workstation pipeline, not a claim of a single-command portable full rerun.

Retain group-level train/validation/test splits. Select gates and penalties on calibration groups only, then evaluate the frozen setting on held-out groups. A rerun must keep candidate-only and full native matcher endpoints separate. MegaDepth archive image-slope proxies are not calibrated surface normals. A paired laser/COLMAP comparison shares other inputs and does not establish annotation equivalence. Test fixtures are not manuscript evidence.
