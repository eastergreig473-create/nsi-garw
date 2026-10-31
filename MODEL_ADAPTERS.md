# Model-adapter protocol

This project is a clean-room reconstruction from the manuscript. It does not claim to recover the authors' undisclosed preprocessing, checkpoint revisions, or layer choices. Every scientific feature export must therefore carry a complete provenance record and must be frozen before the main comparison is run.

## Canonical export

Every adapter writes the common NPZ contract documented in `README.md`. Its `metadata_json` must contain:

- `model`
- `implementation` (repository URL or installed package)
- `implementation_commit`
- `checkpoint` and `checkpoint_sha256`
- `layer`
- `preprocessing`
- `coordinate_system`

Weights must never be downloaded silently during a scientific run. Download, checksum, and register them before feature export. A change in any field creates a new experiment identifier.

The built-in ResNet-50 adapter follows this rule: it accepts only an explicit local checkpoint path, uses the frozen torchvision preprocessing declaration, and maps feature-cell centres back to zero-based coordinates in the original image. The registered `IMAGENET1K_V2` checkpoint is listed in `manifests/asset_registry.csv`; its license-review flag intentionally remains `no` until the author confirms the research-use terms.

## Declared model mapping

| Model | Representation to export | Implementation status |
|---|---|---|
| SIFT | 128-D descriptor at retained keypoints | Built-in optional OpenCV adapter |
| SuperPoint | Descriptor tensor before matching | Built-in pinned LightGlue-repository adapter; license confirmation required |
| XFeat | Official sparse descriptor output | Built-in pinned-repository adapter |
| DeDoDe | Descriptor-B output at DeDoDe-L-v2 detections | Built-in pinned-repository adapter |
| LoFTR | Declared coarse and/or fine tensor before matching | Built-in pinned Kornia adapter |
| RoMa | Declared scale-1/2/4/8/16 encoder feature tensor before matching | Built-in pinned-repository adapter |
| DINOv2 | ViT-S/14 patch tokens at a pinned transformer block | Built-in pinned-repository adapter |
| VGGT | Cached layer-4/11/17/23 frame/global aggregator patch tokens | Built-in pinned-repository adapter |
| MASt3R | Declared single-image CroCo encoder block tokens | Built-in pinned-repository adapter |
| DUSt3R | Declared single-image CroCo encoder block tokens; auxiliary only | Built-in pinned-repository adapter |
| ResNet-50 | Pinned torchvision residual-block tensor; diagnostic only | Built-in pinned-checkpoint adapter (`maxpool`, `layer1`–`layer4`) |

The external-exporter designation is intentional: these projects expose different coordinate systems and checkpoint families, so a generic wrapper could silently compare non-equivalent tensors. The central pipeline accepts their validated NPZ exports and applies identical splits, probes, NSI definitions, and statistics.

## Upstream references

- LightGlue/SuperPoint interface: <https://github.com/cvg/LightGlue/blob/main/README.md>
- Glue Factory training and evaluation framework: <https://github.com/cvg/glue-factory/blob/main/README.md>
- XFeat implementation: <https://github.com/verlab/accelerated_features/blob/main/modules/xfeat.py>
- DINOv2 implementation: <https://github.com/facebookresearch/dinov2>
- DeDoDe implementation: <https://github.com/Parskatt/DeDoDe>
- LoFTR/Kornia implementation: <https://kornia.readthedocs.io/en/latest/feature.html>

The built-in XFeat adapter uses the repository-native sparse extractor, an
explicit local `xfeat.pt`, and original-image coordinates. The built-in DINOv2
adapter uses an explicit local ViT-S/14 checkpoint, exports normalized patch
tokens from a declared block, and maps the aspect-preserving 14-pixel grid back
to the original image. Neither adapter is permitted to download weights during
a scientific run.

The built-in DeDoDe adapter freezes the v2 L detector and B descriptor as two
separately hashed local checkpoints. The LoFTR adapter exports either the
coarse or fine ResNet-FPN tensor before any cross-image transformer or matching
operation, so the declared single-image representation is not contaminated by
the paired target image.

The built-in RoMa adapter loads both the official outdoor checkpoint and the
explicit local DINOv2 ViT-L/14 dependency without network access. It exports a
declared encoder-pyramid scale before cross-image decoding or matching, uses
the repository's square resize and ImageNet normalization, and maps the feature
grid back to the original image coordinate system.

The built-in VGGT adapter instantiates only the official aggregator, loads its
weights from the explicit local VGGT-1B checkpoint, and exports a declared
cached layer's concatenated frame/global patch tokens. Official black
center-padding is reproduced, padded patch positions are removed, and retained
patch centres are mapped back to the original image.

The MASt3R/DUSt3R adapter reconstructs the exact architecture declared inside
each official checkpoint, then executes only the single-image patch embedding
and a declared encoder block. It reproduces the official long-edge resize,
multiple-of-16 centre crop, and `[-1,1]` normalization; no paired decoder or
3D prediction head is evaluated for NSI feature export.

Repository URL, commit, checkpoint identifier, checksum, license, preprocessing, and layer must be copied into each export's metadata. These links are implementation references, not evidence that the current manuscript used the same versions.

## Acceptance checks

Before an export enters the main manifest:

1. Validate array shapes and finite values with `validate-npz`.
2. Confirm that `groups` are scene identifiers, not image identifiers.
3. Confirm that geometry and appearance targets are sampled at exactly the feature coordinates.
4. Check the checkpoint SHA-256 and repository commit against the run registry.
5. Inspect at least ten image overlays for coordinate and resizing errors.
6. Freeze the manifest; changes after freezing require a new run ID.

GARW projectors live in standardized feature coordinates. Before applying `U_G` or `U_A_perp`, adapters must transform both members of every candidate pair with the `feature_mean` and `feature_scale` stored in the corresponding projector NPZ. Mixing raw and standardized coordinates is invalid.
