# Dataset and model asset registry

Checked against author-maintained sources on 2026-10-23. This file records acquisition sources; it does not pin a scientific run. At acquisition time, record the actual repository commit and SHA-256 of every archive/checkpoint in `manifests/asset_registry_template.csv`.

## Datasets

| Asset | Author-maintained source | Size/license note | Required role |
|---|---|---|---|
| MegaDepth v1 | <https://www.cs.cornell.edu/projects/megadepth/> | The official page lists 196 COLMAP locations and a 199 GB compressed v1 archive under CC BY 4.0 for derived data; source images retain their own licenses. The full archive plus extraction is too large to fetch casually into this workspace. | Primary depth/normal probing and cross-scene split |
| ETH3D | <https://eth3d.ethz.ch/> | Official site; data are CC BY-NC-SA 4.0. The site provides training depth maps/laser-derived geometry for supported subsets. | Cross-scene evaluation and independent normal validation |
| HPatches full sequences | <https://github.com/hpatches/hpatches-dataset> | Official repository links the 1.3 GB full sequences and homographies; dataset repository is MIT licensed while source-image citations/licenses must still be respected. | Illumination/viewpoint analysis and homography proxy |

Do not start a full MegaDepth download until the destination has enough space for both the 199 GB archive and extracted data. Prefer a frozen, scene-complete subset for development; use the full registered set for final claims.

## Models and feature implementations

| Model | Author-maintained implementation | Important pinning/license note |
|---|---|---|
| SIFT | <https://docs.opencv.org/> | Record OpenCV build/version and SIFT parameters. |
| SuperPoint | <https://github.com/magicleap/SuperPointPretrainedNetwork> | Original inference/weights repository; its license is more restrictive than Apache-2.0. Do not substitute a reimplementation without labeling it. |
| LightGlue evaluation wrapper | <https://github.com/cvg/LightGlue> and <https://github.com/cvg/glue-factory> | Useful for SuperPoint/SIFT evaluation; LightGlue explicitly notes the separate SuperPoint license. |
| XFeat | <https://github.com/verlab/accelerated_features> | Official code includes 64-D sparse/dense descriptors and weights. |
| DeDoDe | <https://github.com/Parskatt/DeDoDe> | Official README points to versioned pretrained-model releases. |
| LoFTR | <https://github.com/zju3dv/LoFTR> | Official code/weights; choose and record indoor/outdoor DS/OT checkpoint explicitly. |
| RoMa | <https://github.com/Parskatt/RoMa> | Official repository warns that current code differs from the original experiment revision; pin the exact commit and checkpoint. |
| DINOv2 | <https://github.com/facebookresearch/dinov2> | Pin architecture, registers/no-registers variant, block, input resolution, and checkpoint; never use a moving `DEFAULT` alias. |
| VGGT | <https://github.com/facebookresearch/vggt> | Checkpoint licenses differ; the official repository distinguishes original non-commercial and newer commercial checkpoints. Record the exact one. |
| MASt3R | <https://github.com/naver/mast3r> | Official code is CC BY-NC-SA 4.0 and uses recursive dependencies. |
| DUSt3R | <https://github.com/naver/dust3r> | Official code is CC BY-NC-SA 4.0; auxiliary analysis only in this protocol. |
| ResNet-50 | <https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html> | Use an explicit enum such as `IMAGENET1K_V2`, not `DEFAULT`; record torchvision version and preprocessing. |

## Admission rule

An asset may enter a scientific manifest only when all of these fields are non-empty:

- local absolute path;
- source URL;
- license reviewed (`yes/no` plus note);
- repository commit or immutable release tag;
- checkpoint/archive identifier;
- SHA-256;
- acquisition date;
- preprocessing and coordinate convention;
- person/process that verified the record.

The pipeline deliberately does not auto-download model weights. Silent downloads make checksum and license review easy to miss and are therefore excluded from the final-run path.
