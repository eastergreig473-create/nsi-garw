# NSI GARW Experimental Pipeline

This package reconstructs the manuscript's revised NSI and GARW experiments as an auditable, configuration-driven pipeline. It deliberately separates the scientific core from model-specific feature export so that every evaluated model can use the same splits, annotations, metric implementation, and statistics.

## What is implemented

- Group-isolated 8:1:1 splitting and held-out multi-output ridge probes.
- Revised `A`, raw supervised-subspace OSR, `B=1-OSR`, KSG dependence, bounded information purity `C`, and `NSI=A*B*C`.
- Geometry and residualized-appearance projectors for GARW.
- Calibrated sigmoid gate, declared empirical conflict penalty, and ablations.
- MegaDepth/ETH3D depth-to-normal and appearance-gradient helpers.
- HPatches homography-consistency structure-tensor proxy.
- Texture, illumination, motion-blur, JPEG, repetitive-pattern, occlusion, and composite degradations.
- Paired bootstrap confidence intervals, Wilcoxon tests, Spearman association, permutation tests, rank-stability analysis, and publication figure generation.
- A synthetic smoke test that checks the implementation without producing manuscript evidence.
- A pinned-checkpoint torchvision ResNet-50 adapter for `maxpool` and `layer1`–`layer4`, with original-image coordinate mapping and no implicit weight download.

## Scientific input contract

Each feature dump is an `.npz` file containing aligned arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `features` | `N x D` | Feature vectors from one declared model and layer |
| `geometry` | `N x G` | Geometry targets, normally normal components or a documented proxy |
| `appearance` | `N x A` | Appearance targets, normally gradient magnitude |
| `groups` | `N` | Scene identifiers used to prevent leakage |
| `split` | `N`, optional | `train`, `validation`, or `test` |
| `sample_ids` | `N`, optional | Image, keypoint, or patch identifiers |

The scientific batch manifest requires `experiment_id`, `model`, `dataset`, `layer`, `condition`, `strength`, `input_npz`, `implementation`, `implementation_commit`, `checkpoint`, `checkpoint_sha256`, `preprocessing`, `coordinate_system`, `annotation_source`, and `split_id`. The model, layer, implementation commit, and checkpoint hash must match the NPZ metadata exactly.

## Local setup

```powershell
cd D:\改论文加实验\nsi_garw_pipeline
D:\改论文加实验\.envs\nsi-garw\python.exe -m pip install -e .
```

Optional model inference dependencies are intentionally separate:

```powershell
D:\改论文加实验\.envs\nsi-garw\python.exe -m pip install -e ".[models]"
```

## Verification without scientific data

```powershell
$env:PYTHONPATH = "D:\改论文加实验\nsi_garw_pipeline\src"
D:\改论文加实验\.envs\nsi-garw\python.exe -m pytest -q
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli smoke --output-dir results\smoke
```

Smoke-test numbers must never be copied into the manuscript.

## Scientific runs

```powershell
$env:PYTHONPATH = "D:\改论文加实验\nsi_garw_pipeline\src"
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli validate-npz --input data\example.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli attributes-depth --image data\image.jpg --depth data\depth.npy --intrinsics 1000 1000 512 384 --output data\attributes.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli attributes-hpatches --reference data\1.ppm --target data\2.ppm --homography data\H_1_2.npy --proxy ncc --output data\hpatches_attributes.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli prepare-hpatches --dataset-root F:\NSI_GARW_Data\datasets\HPatches\hpatches-sequences-release --output-dir F:\NSI_GARW_Data\manifests\hpatches --seed 20261023
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-roma --image data\1.ppm --repository external\roma --checkpoint F:\NSI_GARW_Data\models\roma_outdoor.pth --dinov2-checkpoint F:\NSI_GARW_Data\models\dinov2_vitl14_pretrain.pth --scale 16 --resolution 560 --device cuda --output results\roma_scale16.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-vggt --image data\1.ppm --repository external\vggt --checkpoint F:\NSI_GARW_Data\models\vggt_1b_model.pt --layer 23 --image-size 518 --device cuda --output results\vggt_layer23.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-mast3r --image data\1.ppm --repository external\mast3r --checkpoint F:\NSI_GARW_Data\models\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --layer 23 --image-size 512 --device cuda --output results\mast3r_encoder23.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-dust3r --image data\1.ppm --repository external\mast3r --checkpoint F:\NSI_GARW_Data\models\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --layer 23 --image-size 512 --device cuda --output results\dust3r_encoder23.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-resnet50 --image data\image.jpg --checkpoint weights\resnet50-11ad3fa6.pth --layer layer4 --device cuda --output data\resnet50_layer4_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-superpoint --image data\image.jpg --repository external\lightglue --checkpoint weights\superpoint_v1.pth --max-keypoints 4096 --resize 1024 --device cuda --output data\superpoint_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-xfeat --image data\image.jpg --repository external\XFeat --checkpoint F:\NSI_GARW_Data\models\xfeat.pt --max-keypoints 4096 --detection-threshold 0.05 --device cuda --output data\xfeat_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-dinov2 --image data\image.jpg --repository external\dinov2 --checkpoint F:\NSI_GARW_Data\models\dinov2_vits14_pretrain.pth --layer 11 --max-size 518 --device cuda --output data\dinov2_block11_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-dedode --image data\image.jpg --repository external\DeDoDe --detector-checkpoint F:\NSI_GARW_Data\models\dedode_detector_L_v2.pth --descriptor-checkpoint F:\NSI_GARW_Data\models\dedode_descriptor_B.pth --max-keypoints 4096 --resize 784 --device cuda --output data\dedode_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli extract-loftr --image data\image.jpg --checkpoint F:\NSI_GARW_Data\models\loftr_outdoor.ckpt --representation coarse --max-size 1024 --device cuda --output data\loftr_coarse_local.npz
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli build-hpatches-model --adapter sift --export-id sift_default --pairs-manifest F:\NSI_GARW_Data\manifests\hpatches\hpatches_pairs.csv --dataset-root F:\NSI_GARW_Data\datasets\HPatches\hpatches-sequences-release --cache-root F:\NSI_GARW_Data\cache --output F:\NSI_GARW_Data\manifests\features\sift_default.npz --max-samples-per-pair 64 --seed 20261023
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli compute-nsi --input data\example.npz --output results\example.json
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli batch-nsi --manifest manifests\nsi.csv --output-dir results\nsi
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli sensitivity --input data\example.npz --output results\sensitivity.csv
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli sensitivity-batch --manifest manifests\sensitivity.csv --output-dir results\sensitivity
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli garw --input data\garw_candidates.csv --output-dir results\garw
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli fit-local-proxy --input data\local_proxy.csv --feature-column texture_energy --feature-column projected_disagreement --output data\garw_candidates.csv --model-output results\local_proxy.json
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli evaluate-matches --input data\matches.csv --output-dir results\matching --primary-method GARW --baseline-method LoFTR --thresholds 1 3 5 10
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli association --input results\association.csv --x-column NSI --y-column robustness --group-column scene --output results\association.json
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli rank-stability --input results\rank_variants.csv --baseline-variant default --output results\rank_stability.csv
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli figures --results-dir results --output-dir figures
```

The full experiment command is frozen below. Execution remains fail-closed until the real datasets, weight identifiers, and model-specific feature exporters are registered. See `MODEL_ADAPTERS.md` and `IMPLEMENTATION_CONTRACT.md` before generating any manuscript result.

## Frozen full-rerun entry point

The final suite has one fail-closed entry point. Without `--execute` it performs
only the asset, license-record, checksum, and manifest preflight:

```powershell
$env:PYTHONPATH = "D:\改论文加实验\nsi_garw_pipeline\src"
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli full-rerun --plan configs\full_rerun.toml --output-dir results\final
```

After every registered archive/checkpoint and scientific input manifest passes,
the exact execution command is:

```powershell
D:\改论文加实验\.envs\nsi-garw\python.exe -m nsi_garw.cli full-rerun --plan configs\full_rerun.toml --output-dir results\final --execute
```

The command refuses a non-empty output directory, missing license review, path or
SHA-256 mismatch, and any template placeholder. It never downloads a model or
dataset silently and never falls back to legacy manuscript values.

