from __future__ import annotations

import numpy as np
import pytest

from nsi_garw.adapters import (
    DeDoDeAdapter,
    DINOv2Adapter,
    LoFTRBackboneAdapter,
    RoMaEncoderAdapter,
    VGGTAggregatorAdapter,
    CroCoEncoderAdapter,
    ResNet50Adapter,
    XFeatAdapter,
)


def test_resnet_coordinate_grid_maps_cell_centres_to_original_image() -> None:
    coordinates = ResNet50Adapter._coordinate_grid(
        original_hw=(100, 200),
        resized_hw=(232, 464),
        crop_hw=(224, 224),
        feature_hw=(2, 2),
    )
    assert coordinates.shape == (4, 2)
    assert np.isfinite(coordinates).all()
    assert np.all(coordinates[:, 0] >= 0)
    assert np.all(coordinates[:, 0] < 200)
    assert np.all(coordinates[:, 1] >= 0)
    assert np.all(coordinates[:, 1] < 100)
    assert coordinates[0, 0] < coordinates[1, 0]
    assert coordinates[0, 1] < coordinates[2, 1]


def test_resnet_adapter_rejects_unknown_layer_before_loading_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unsupported ResNet-50 layer"):
        ResNet50Adapter("missing.pth", layer="not-a-layer", device="cpu")


def test_dinov2_coordinate_grid_covers_original_image() -> None:
    coordinates = DINOv2Adapter._coordinate_grid(
        original_hw=(140, 280),
        feature_hw=(10, 20),
    )
    assert coordinates.shape == (200, 2)
    assert np.isfinite(coordinates).all()
    assert np.all(coordinates[:, 0] > 0)
    assert np.all(coordinates[:, 0] < 280)
    assert np.all(coordinates[:, 1] > 0)
    assert np.all(coordinates[:, 1] < 140)


def test_dinov2_adapter_rejects_unknown_layer_before_loading_checkpoint() -> None:
    with pytest.raises(ValueError, match="layer must be in"):
        DINOv2Adapter("missing-repository", "missing.pth", layer=12, device="cpu")


def test_xfeat_adapter_rejects_invalid_threshold_before_loading_checkpoint() -> None:
    with pytest.raises(ValueError, match="detection_threshold"):
        XFeatAdapter(
            "missing-repository",
            "missing.pth",
            detection_threshold=1.5,
            device="cpu",
        )


def test_dedode_adapter_rejects_invalid_resize_before_loading_checkpoint() -> None:
    with pytest.raises(ValueError, match="resize must be positive"):
        DeDoDeAdapter(
            "missing-repository",
            "missing-detector.pth",
            "missing-descriptor.pth",
            resize=0,
            device="cpu",
        )


def test_loftr_adapter_rejects_unknown_representation_before_checkpoint() -> None:
    with pytest.raises(ValueError, match="representation must be one of"):
        LoFTRBackboneAdapter(
            "missing.ckpt", representation="unknown", device="cpu"
        )


def test_roma_adapter_rejects_unknown_scale_before_checkpoints() -> None:
    with pytest.raises(ValueError, match="scale must be one of"):
        RoMaEncoderAdapter(
            "missing-repository",
            "missing-roma.pth",
            "missing-dinov2.pth",
            scale=3,
            device="cpu",
        )


def test_roma_adapter_requires_resolution_divisible_by_fourteen() -> None:
    with pytest.raises(ValueError, match="multiple of 14"):
        RoMaEncoderAdapter(
            "missing-repository",
            "missing-roma.pth",
            "missing-dinov2.pth",
            resolution=512,
            device="cpu",
        )


def test_vggt_adapter_rejects_uncached_layer_before_checkpoint() -> None:
    with pytest.raises(ValueError, match="layer must be one of"):
        VGGTAggregatorAdapter(
            "missing-repository", "missing.pt", layer=5, device="cpu"
        )


def test_vggt_adapter_requires_image_size_divisible_by_fourteen() -> None:
    with pytest.raises(ValueError, match="multiple of 14"):
        VGGTAggregatorAdapter(
            "missing-repository", "missing.pt", image_size=512, device="cpu"
        )


def test_croco_adapter_rejects_unknown_model_before_checkpoint() -> None:
    with pytest.raises(ValueError, match="model_type must be one of"):
        CroCoEncoderAdapter(
            "missing-repository",
            "missing.pth",
            model_type="unknown",
            device="cpu",
        )


def test_croco_adapter_rejects_negative_layer_before_checkpoint() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CroCoEncoderAdapter(
            "missing-repository",
            "missing.pth",
            model_type="mast3r",
            layer=-1,
            device="cpu",
        )
