from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any, Protocol

import numpy as np

from .io import FeatureDataset, load_feature_npz


@dataclass(frozen=True)
class LocalFeatures:
    coordinates_xy: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        coordinates = np.asarray(self.coordinates_xy)
        descriptors = np.asarray(self.descriptors)
        scores = np.asarray(self.scores)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates_xy must have shape N x 2")
        if descriptors.ndim != 2 or descriptors.shape[0] != coordinates.shape[0]:
            raise ValueError("descriptors must have shape N x D")
        if scores.ndim != 1 or scores.shape[0] != coordinates.shape[0]:
            raise ValueError("scores must have shape N")
        if not (
            np.isfinite(coordinates).all()
            and np.isfinite(descriptors).all()
            and np.isfinite(scores).all()
        ):
            raise ValueError("Local features contain non-finite values")


class LocalFeatureAdapter(Protocol):
    name: str

    def extract(self, image: np.ndarray) -> LocalFeatures: ...


def _repository_revision(repository: Path) -> str:
    """Return a Git commit when available, otherwise a content-addressed snapshot ID."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        files = sorted(
            path
            for path in repository.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix.lower() not in {".pyc", ".pyo"}
        )
        for path in files:
            relative = path.relative_to(repository).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return f"snapshot-sha256:{digest.hexdigest()}"


class NPZFeatureAdapter:
    """The canonical adapter for model-independent NSI analysis."""

    name = "npz"

    @staticmethod
    def load(path: str | Path) -> FeatureDataset:
        return load_feature_npz(path)


class SIFTAdapter:
    name = "SIFT"

    def __init__(self, max_keypoints: int = 4096) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "SIFTAdapter requires opencv-contrib-python from the models extra"
            ) from error
        self.cv2 = cv2
        self.max_keypoints = int(max_keypoints)
        self.implementation_version = f"opencv-{cv2.__version__}"
        self.extractor = cv2.SIFT_create(nfeatures=self.max_keypoints)

    def _metadata(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "implementation": "OpenCV SIFT",
            "implementation_commit": self.implementation_version,
            "checkpoint": "not-applicable",
            "checkpoint_sha256": "not-applicable",
            "layer": "128-D descriptor at detected keypoints",
            "preprocessing": (
                f"RGB converted to OpenCV grayscale uint8; nfeatures {self.max_keypoints}"
            ),
            "coordinate_system": "zero-based original-image x,y pixels",
        }

    def extract(self, image: np.ndarray) -> LocalFeatures:
        value = np.asarray(image)
        if value.ndim == 3:
            if value.shape[2] < 3:
                value = value[..., 0]
            else:
                value = self.cv2.cvtColor(value[..., :3], self.cv2.COLOR_RGB2GRAY)
        if value.dtype != np.uint8:
            float_value = value.astype(np.float64)
            if float_value.max(initial=0.0) <= 1.0:
                float_value *= 255.0
            value = np.clip(np.rint(float_value), 0, 255).astype(np.uint8)
        keypoints, descriptors = self.extractor.detectAndCompute(value, None)
        if not keypoints or descriptors is None:
            return LocalFeatures(
                coordinates_xy=np.zeros((0, 2), dtype=np.float64),
                descriptors=np.zeros((0, 128), dtype=np.float64),
                scores=np.zeros(0, dtype=np.float64),
                metadata=self._metadata(),
            )
        order = np.argsort([-point.response for point in keypoints])[: self.max_keypoints]
        coordinates = np.asarray([keypoints[index].pt for index in order], dtype=np.float64)
        scores = np.asarray([keypoints[index].response for index in order], dtype=np.float64)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=np.asarray(descriptors[order], dtype=np.float64),
            scores=scores,
            metadata=self._metadata(),
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class ResNet50Adapter:
    """Pinned torchvision ResNet-50 feature-map exporter.

    The adapter never downloads weights.  It loads an explicitly registered
    checkpoint and maps feature-cell centres back to coordinates in the
    original input image, allowing target maps to be sampled without silently
    mixing coordinate systems.
    """

    name = "ResNet-50"
    _VALID_LAYERS = {"maxpool", "layer1", "layer2", "layer3", "layer4"}

    def __init__(
        self,
        checkpoint: str | Path,
        layer: str = "layer4",
        device: str | None = None,
    ) -> None:
        if layer not in self._VALID_LAYERS:
            raise ValueError(f"Unsupported ResNet-50 layer {layer!r}")
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import torch
            from torchvision.models import resnet50
            from torchvision.models.feature_extraction import create_feature_extractor
        except ImportError as error:
            raise RuntimeError(
                "ResNet50Adapter requires torch and torchvision from the models extra"
            ) from error

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet50(weights=None)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.extractor = create_feature_extractor(model, return_nodes={layer: "features"})
        self.extractor.to(selected_device)
        self.torch = torch
        self.device = torch.device(selected_device)
        self.layer = layer
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    @staticmethod
    def _coordinate_grid(
        original_hw: tuple[int, int],
        resized_hw: tuple[int, int],
        crop_hw: tuple[int, int],
        feature_hw: tuple[int, int],
    ) -> np.ndarray:
        original_h, original_w = original_hw
        resized_h, resized_w = resized_hw
        crop_h, crop_w = crop_hw
        feature_h, feature_w = feature_hw
        crop_top = max((resized_h - crop_h) // 2, 0)
        crop_left = max((resized_w - crop_w) // 2, 0)
        x_resized = crop_left + (np.arange(feature_w, dtype=np.float64) + 0.5) * (
            crop_w / feature_w
        )
        y_resized = crop_top + (np.arange(feature_h, dtype=np.float64) + 0.5) * (
            crop_h / feature_h
        )
        x_original = x_resized * (original_w / resized_w)
        y_original = y_resized * (original_h / resized_h)
        grid_x, grid_y = np.meshgrid(x_original, y_original)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def extract(self, image: np.ndarray) -> LocalFeatures:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as vision_functional

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3:
            raise ValueError("image must be H x W or H x W x C")
        if value.shape[2] == 1:
            value = np.repeat(value, 3, axis=2)
        if value.shape[2] < 3:
            raise ValueError("image must have at least one or three channels")
        value = value[..., :3]
        original_h, original_w = value.shape[:2]
        tensor = self.torch.from_numpy(np.array(value, copy=True, order="C")).to(
            device=self.device, dtype=self.torch.float32
        )
        tensor = tensor.permute(2, 0, 1)
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = vision_functional.resize(
            tensor, 232, interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized_h, resized_w = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        tensor = vision_functional.center_crop(tensor, [224, 224])
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0)
        with self.torch.inference_mode():
            feature_map = self.extractor(tensor)["features"][0]
        descriptors = feature_map.permute(1, 2, 0).float().cpu().numpy()
        feature_h, feature_w = descriptors.shape[:2]
        descriptors = descriptors.reshape(feature_h * feature_w, -1).astype(np.float64)
        coordinates = self._coordinate_grid(
            (original_h, original_w),
            (resized_h, resized_w),
            (224, 224),
            (feature_h, feature_w),
        )
        scores = np.linalg.norm(descriptors, axis=1)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": self.name,
                "implementation": "torchvision",
                "implementation_commit": "torchvision-0.20.1+cu121",
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": self.layer,
                "preprocessing": (
                    "RGB; shorter edge 232 bilinear antialias; center crop 224; "
                    "ImageNet mean/std; FP32"
                ),
                "coordinate_system": (
                    "feature-cell centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class SuperPointAdapter:
    """Pinned LightGlue-repository SuperPoint exporter with no network access."""

    name = "SuperPoint"
    _EXPECTED_WEIGHT_URL = (
        "https://github.com/cvg/LightGlue/releases/download/"
        "v0.1_arxiv/superpoint_v1.pth"
    )

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        max_keypoints: int = 4096,
        resize: int = 1024,
        device: str | None = None,
    ) -> None:
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (repository_path / "lightglue" / "superpoint.py").is_file():
            raise FileNotFoundError(
                f"LightGlue SuperPoint source not found under {repository_path}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        if resize <= 0:
            raise ValueError("resize must be positive")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("SuperPointAdapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from lightglue.superpoint import SuperPoint

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        original_loader = torch.hub.load_state_dict_from_url

        def local_weight_loader(url: str, *args: Any, **kwargs: Any) -> Any:
            if url != self._EXPECTED_WEIGHT_URL:
                raise RuntimeError(f"Refusing undeclared weight URL: {url}")
            return state

        torch.hub.load_state_dict_from_url = local_weight_loader
        try:
            model = SuperPoint(max_num_keypoints=int(max_keypoints))
        finally:
            torch.hub.load_state_dict_from_url = original_loader

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(selected_device)
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.max_keypoints = int(max_keypoints)
        self.resize = int(resize)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = value[..., :3]
        tensor = self.torch.from_numpy(np.array(value, copy=True, order="C")).to(
            device=self.device, dtype=self.torch.float32
        )
        tensor = tensor.permute(2, 0, 1)
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        with self.torch.inference_mode():
            output = self.model.extract(tensor, resize=self.resize)
        coordinates = output["keypoints"][0].float().cpu().numpy().astype(np.float64)
        descriptors = output["descriptors"][0].float().cpu().numpy().astype(np.float64)
        scores = output["keypoint_scores"][0].float().cpu().numpy().astype(np.float64)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": self.name,
                "implementation": "https://github.com/cvg/LightGlue.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": "descriptor tensor before matching",
                "preprocessing": (
                    f"RGB float [0,1]; LightGlue SuperPoint longest-edge resize {self.resize}; "
                    f"max_keypoints {self.max_keypoints}; FP32"
                ),
                "coordinate_system": "zero-based original-image x,y pixels",
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class XFeatAdapter:
    """Pinned XFeat sparse descriptor exporter with explicit local weights."""

    name = "XFeat"

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        max_keypoints: int = 4096,
        detection_threshold: float = 0.05,
        device: str | None = None,
    ) -> None:
        if max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in [0, 1]")
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (repository_path / "modules" / "xfeat.py").is_file():
            raise FileNotFoundError(f"XFeat source not found under {repository_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("XFeatAdapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from modules.xfeat import XFeat

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = XFeat(
            weights=state,
            top_k=int(max_keypoints),
            detection_threshold=float(detection_threshold),
        )
        model.dev = torch.device(selected_device)
        model.net.to(model.dev).eval()
        model.to(model.dev).eval()
        self.model = model
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.max_keypoints = int(max_keypoints)
        self.detection_threshold = float(detection_threshold)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        tensor = self.torch.from_numpy(value).permute(2, 0, 1)[None].to(
            device=self.device, dtype=self.torch.float32
        )
        with self.torch.inference_mode():
            output = self.model.detectAndCompute(
                tensor,
                top_k=self.max_keypoints,
                detection_threshold=self.detection_threshold,
            )[0]
        coordinates = output["keypoints"].float().cpu().numpy().astype(np.float64)
        descriptors = output["descriptors"].float().cpu().numpy().astype(np.float64)
        scores = output["scores"].float().cpu().numpy().astype(np.float64)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": self.name,
                "implementation": "https://github.com/verlab/accelerated_features.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": "official sparse descriptor output",
                "preprocessing": (
                    "RGB FP32 in repository-native intensity scale; bilinear resize to "
                    "dimensions divisible by 32; "
                    f"top_k {self.max_keypoints}; detection_threshold {self.detection_threshold:g}"
                ),
                "coordinate_system": "zero-based original-image x,y pixels",
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class DINOv2Adapter:
    """Pinned DINOv2 ViT-S/14 patch-token exporter using local source and weights."""

    name = "DINOv2"
    _PATCH_SIZE = 14
    _BLOCK_COUNT = 12

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        layer: int = 11,
        max_size: int = 518,
        device: str | None = None,
    ) -> None:
        if not 0 <= int(layer) < self._BLOCK_COUNT:
            raise ValueError(
                f"DINOv2 ViT-S/14 layer must be in [0, {self._BLOCK_COUNT - 1}]"
            )
        if max_size < self._PATCH_SIZE:
            raise ValueError(f"max_size must be at least {self._PATCH_SIZE}")
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (repository_path / "dinov2" / "hub" / "backbones.py").is_file():
            raise FileNotFoundError(f"DINOv2 source not found under {repository_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv2Adapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from dinov2.hub.backbones import dinov2_vits14

        model = dinov2_vits14(pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(selected_device)
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.layer = int(layer)
        self.max_size = int(max_size)

    @classmethod
    def _resized_shape(cls, original_hw: tuple[int, int], max_size: int) -> tuple[int, int]:
        height, width = original_hw
        scale = float(max_size) / float(max(height, width))
        resized_h = max(cls._PATCH_SIZE, int(round(height * scale / cls._PATCH_SIZE)) * cls._PATCH_SIZE)
        resized_w = max(cls._PATCH_SIZE, int(round(width * scale / cls._PATCH_SIZE)) * cls._PATCH_SIZE)
        return resized_h, resized_w

    @staticmethod
    def _coordinate_grid(
        original_hw: tuple[int, int], feature_hw: tuple[int, int]
    ) -> np.ndarray:
        original_h, original_w = original_hw
        feature_h, feature_w = feature_hw
        x = (np.arange(feature_w, dtype=np.float64) + 0.5) * (original_w / feature_w)
        y = (np.arange(feature_h, dtype=np.float64) + 0.5) * (original_h / feature_h)
        grid_x, grid_y = np.meshgrid(x, y)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def extract(self, image: np.ndarray) -> LocalFeatures:
        import torch.nn.functional as functional

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        original_h, original_w = value.shape[:2]
        resized_h, resized_w = self._resized_shape(
            (original_h, original_w), self.max_size
        )
        tensor = self.torch.from_numpy(value).permute(2, 0, 1)[None].to(
            device=self.device, dtype=self.torch.float32
        )
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        tensor = (tensor - mean) / std
        with self.torch.inference_mode():
            feature_map = self.model.get_intermediate_layers(
                tensor,
                n=[self.layer],
                reshape=True,
                return_class_token=False,
                norm=True,
            )[0][0]
        descriptors = feature_map.permute(1, 2, 0).float().cpu().numpy()
        feature_h, feature_w = descriptors.shape[:2]
        descriptors = descriptors.reshape(feature_h * feature_w, -1).astype(np.float64)
        coordinates = self._coordinate_grid(
            (original_h, original_w), (feature_h, feature_w)
        )
        scores = np.linalg.norm(descriptors, axis=1)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": "DINOv2 ViT-S/14",
                "implementation": "https://github.com/facebookresearch/dinov2.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": f"blocks.{self.layer} normalized patch tokens",
                "preprocessing": (
                    f"RGB float [0,1]; aspect-preserving bilinear resize with long edge "
                    f"approximately {self.max_size} and both dimensions divisible by 14; "
                    "ImageNet mean/std; FP32"
                ),
                "coordinate_system": (
                    "patch-cell centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class DeDoDeAdapter:
    """Pinned DeDoDe-L-v2 detector and descriptor-B exporter."""

    name = "DeDoDe"

    def __init__(
        self,
        repository: str | Path,
        detector_checkpoint: str | Path,
        descriptor_checkpoint: str | Path,
        *,
        max_keypoints: int = 4096,
        resize: int = 784,
        device: str | None = None,
    ) -> None:
        if max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        if resize <= 0:
            raise ValueError("resize must be positive")
        repository_path = Path(repository).resolve()
        detector_path = Path(detector_checkpoint).resolve()
        descriptor_path = Path(descriptor_checkpoint).resolve()
        if not (repository_path / "DeDoDe" / "model_zoo" / "dedode_models.py").is_file():
            raise FileNotFoundError(f"DeDoDe source not found under {repository_path}")
        if not detector_path.is_file():
            raise FileNotFoundError(detector_path)
        if not descriptor_path.is_file():
            raise FileNotFoundError(descriptor_path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DeDoDeAdapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from DeDoDe import dedode_descriptor_B, dedode_detector_L

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        detector_state = torch.load(detector_path, map_location="cpu", weights_only=True)
        descriptor_state = torch.load(descriptor_path, map_location="cpu", weights_only=True)
        if isinstance(detector_state, dict) and "state_dict" in detector_state:
            detector_state = detector_state["state_dict"]
        if isinstance(descriptor_state, dict) and "state_dict" in descriptor_state:
            descriptor_state = descriptor_state["state_dict"]
        self.detector = dedode_detector_L(
            device=selected_device, weights=detector_state, remove_borders=False
        ).eval()
        self.descriptor = dedode_descriptor_B(
            device=selected_device, weights=descriptor_state
        ).eval()
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.detector_checkpoint = detector_path
        self.descriptor_checkpoint = descriptor_path
        self.detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
        self.descriptor_sha256 = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        self.max_keypoints = int(max_keypoints)
        self.resize = int(resize)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        import torch.nn.functional as functional

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        original_h, original_w = value.shape[:2]
        tensor = self.torch.from_numpy(value).permute(2, 0, 1)[None].to(
            device=self.device, dtype=self.torch.float32
        )
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        batch = {"image": (tensor - mean) / std}
        with self.torch.inference_mode():
            detected = self.detector.detect(batch, num_keypoints=self.max_keypoints)
            keypoints_normalized = detected["keypoints"]
            described = self.descriptor.describe_keypoints(
                batch, keypoints_normalized
            )["descriptions"]
        keypoints = self.detector.to_pixel_coords(
            keypoints_normalized, original_h, original_w
        )
        coordinates = keypoints[0].float().cpu().numpy().astype(np.float64)
        descriptors = described[0].float().cpu().numpy().astype(np.float64)
        scores = detected["confidence"][0].float().cpu().numpy().astype(np.float64)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": "DeDoDe-L-v2 + descriptor-B",
                "implementation": "https://github.com/Parskatt/DeDoDe.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": (
                    f"detector={self.detector_checkpoint}; "
                    f"descriptor={self.descriptor_checkpoint}"
                ),
                "checkpoint_sha256": (
                    f"detector={self.detector_sha256};descriptor={self.descriptor_sha256}"
                ),
                "layer": "official descriptor-B output at DeDoDe-L-v2 detections",
                "preprocessing": (
                    f"RGB float [0,1]; bilinear square resize {self.resize}; "
                    "ImageNet mean/std; repository AMP policy; "
                    f"top_k {self.max_keypoints}"
                ),
                "coordinate_system": "zero-based original-image x,y pixels",
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class LoFTRBackboneAdapter:
    """Pinned Kornia LoFTR pre-matching backbone exporter."""

    name = "LoFTR"
    _VALID_REPRESENTATIONS = {"coarse", "fine"}

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        representation: str = "coarse",
        max_size: int = 1024,
        device: str | None = None,
    ) -> None:
        if representation not in self._VALID_REPRESENTATIONS:
            raise ValueError(
                f"LoFTR representation must be one of {sorted(self._VALID_REPRESENTATIONS)}"
            )
        if max_size < 8:
            raise ValueError("max_size must be at least 8")
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import kornia
            import torch
            from kornia.feature import LoFTR
        except ImportError as error:
            raise RuntimeError("LoFTRBackboneAdapter requires torch and kornia") from error

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = LoFTR(pretrained=None)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(selected_device)
        self.torch = torch
        self.device = torch.device(selected_device)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.implementation_version = f"kornia-{kornia.__version__}"
        self.representation = representation
        self.max_size = int(max_size)

    @staticmethod
    def _resized_shape(original_hw: tuple[int, int], max_size: int) -> tuple[int, int]:
        height, width = original_hw
        scale = float(max_size) / float(max(height, width))
        resized_h = max(8, int(round(height * scale / 8.0)) * 8)
        resized_w = max(8, int(round(width * scale / 8.0)) * 8)
        return resized_h, resized_w

    def extract(self, image: np.ndarray) -> LocalFeatures:
        import torch.nn.functional as functional

        value = np.asarray(image)
        if value.ndim == 3:
            value = value[..., :3].astype(np.float64).mean(axis=2)
        if value.ndim != 2:
            raise ValueError("image must be H x W or H x W x C")
        original_h, original_w = value.shape
        resized_h, resized_w = self._resized_shape(
            (original_h, original_w), self.max_size
        )
        tensor = self.torch.from_numpy(np.array(value, copy=True, order="C"))[None, None].to(
            device=self.device, dtype=self.torch.float32
        )
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        with self.torch.inference_mode():
            coarse, fine = self.model.backbone(tensor)
        feature_map = coarse[0] if self.representation == "coarse" else fine[0]
        descriptors = feature_map.permute(1, 2, 0).float().cpu().numpy()
        feature_h, feature_w = descriptors.shape[:2]
        descriptors = descriptors.reshape(feature_h * feature_w, -1).astype(np.float64)
        coordinates = DINOv2Adapter._coordinate_grid(
            (original_h, original_w), (feature_h, feature_w)
        )
        scores = np.linalg.norm(descriptors, axis=1)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": "LoFTR outdoor",
                "implementation": "kornia.feature.LoFTR",
                "implementation_commit": self.implementation_version,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": f"ResNetFPN {self.representation} feature map before matching",
                "preprocessing": (
                    f"grayscale mean of RGB; float [0,1]; aspect-preserving bilinear "
                    f"resize with long edge approximately {self.max_size} and dimensions "
                    "divisible by 8; FP32"
                ),
                "coordinate_system": (
                    "feature-cell centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class RoMaEncoderAdapter:
    """Pinned RoMa encoder feature-pyramid exporter with local weights only."""

    name = "RoMa"
    _VALID_SCALES = {1, 2, 4, 8, 16}

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        dinov2_checkpoint: str | Path,
        *,
        scale: int = 16,
        resolution: int = 560,
        device: str | None = None,
    ) -> None:
        if int(scale) not in self._VALID_SCALES:
            raise ValueError(
                f"RoMa scale must be one of {sorted(self._VALID_SCALES)}"
            )
        if resolution <= 0 or resolution % 14 != 0:
            raise ValueError("RoMa resolution must be a positive multiple of 14")
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        dinov2_path = Path(dinov2_checkpoint).resolve()
        if not (repository_path / "romatch" / "models" / "encoders.py").is_file():
            raise FileNotFoundError(f"RoMa source not found under {repository_path}")
        for path in (checkpoint_path, dinov2_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("RoMaEncoderAdapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from romatch import roma_outdoor

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("highest")
        roma_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        dinov2_state = torch.load(dinov2_path, map_location="cpu", weights_only=True)
        matcher = roma_outdoor(
            device="cpu",
            weights=roma_state,
            dinov2_weights=dinov2_state,
            coarse_res=int(resolution),
            upsample_res=int(resolution),
            amp_dtype=torch.float16,
            symmetric=False,
            use_custom_corr=False,
            upsample_preds=False,
        )
        # RoMa overrides ``train`` and returns the inner VGG module, so retain
        # the composite encoder explicitly instead of assigning ``eval()``.
        encoder = matcher.encoder
        encoder.eval()
        del matcher, roma_state, dinov2_state
        encoder.cnn.to(selected_device).eval()
        encoder.amp = selected_device.startswith("cuda")
        encoder.amp_dtype = torch.float16 if encoder.amp else torch.float32
        # The upstream encoder holds DINOv2 in a plain list, so moving the
        # enclosing module does not move it. Keep its weights and inputs in
        # the same explicit device/dtype even when the device remains CPU.
        encoder.dinov2_vitl14[0] = encoder.dinov2_vitl14[0].to(
            device=selected_device, dtype=encoder.amp_dtype
        ).eval()
        self.encoder = encoder
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.dinov2_checkpoint = dinov2_path
        self.dinov2_sha256 = hashlib.sha256(dinov2_path.read_bytes()).hexdigest()
        self.scale = int(scale)
        self.resolution = int(resolution)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        import torch.nn.functional as functional

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        original_h, original_w = value.shape[:2]
        tensor = self.torch.from_numpy(value).permute(2, 0, 1)[None].to(
            device=self.device, dtype=self.torch.float32
        )
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = self.torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=tensor.dtype
        )[None, :, None, None]
        std = self.torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=tensor.dtype
        )[None, :, None, None]
        tensor = (tensor - mean) / std
        with self.torch.inference_mode():
            pyramid = self.encoder(tensor, upsample=False)
        feature_map = pyramid[self.scale][0]
        descriptors = feature_map.permute(1, 2, 0).float().cpu().numpy()
        feature_h, feature_w = descriptors.shape[:2]
        descriptors = descriptors.reshape(feature_h * feature_w, -1).astype(np.float64)
        coordinates = DINOv2Adapter._coordinate_grid(
            (original_h, original_w), (feature_h, feature_w)
        )
        scores = np.linalg.norm(descriptors, axis=1)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": "RoMa outdoor",
                "implementation": "https://github.com/Parskatt/RoMa.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": (
                    f"roma={self.checkpoint}; dinov2={self.dinov2_checkpoint}"
                ),
                "checkpoint_sha256": (
                    f"roma={self.checkpoint_sha256};dinov2={self.dinov2_sha256}"
                ),
                "layer": f"encoder feature pyramid scale {self.scale}",
                "preprocessing": (
                    f"RGB float [0,1]; official square bilinear resize "
                    f"{self.resolution}x{self.resolution}; ImageNet mean/std; "
                    "repository AMP policy" + ("; CPU FP32 execution" if self.device.type == "cpu" else "")
                ),
                "coordinate_system": (
                    "feature-cell centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class VGGTAggregatorAdapter:
    """Pinned VGGT aggregator-token exporter using an explicit local checkpoint."""

    name = "VGGT"
    _CACHED_LAYERS = {4, 11, 17, 23}
    _PATCH_SIZE = 14

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        layer: int = 23,
        image_size: int = 518,
        device: str | None = None,
    ) -> None:
        if int(layer) not in self._CACHED_LAYERS:
            raise ValueError(
                f"VGGT layer must be one of {sorted(self._CACHED_LAYERS)}"
            )
        if image_size <= 0 or image_size % self._PATCH_SIZE != 0:
            raise ValueError("VGGT image_size must be a positive multiple of 14")
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (repository_path / "vggt" / "models" / "aggregator.py").is_file():
            raise FileNotFoundError(f"VGGT source not found under {repository_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("VGGTAggregatorAdapter requires torch") from error

        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        from vggt.models.aggregator import Aggregator

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("VGGT checkpoint must contain a state dictionary")
        aggregator_state = {
            key.removeprefix("aggregator."): value
            for key, value in state.items()
            if key.startswith("aggregator.")
        }
        if not aggregator_state:
            aggregator_state = dict(state)
        # The pinned VGGT constructor converts a CPU-only stochastic-depth
        # schedule to Python scalars with ``Tensor.item()``.  Parameter
        # allocation can still happen on the meta device, but that small
        # schedule must remain on CPU because meta tensors have no values.
        original_linspace = torch.linspace

        def _cpu_linspace(*args: Any, **kwargs: Any):
            kwargs.setdefault("device", "cpu")
            return original_linspace(*args, **kwargs)

        torch.linspace = _cpu_linspace
        try:
            with torch.device("meta"):
                model = Aggregator(img_size=int(image_size))
        finally:
            torch.linspace = original_linspace
        model.load_state_dict(aggregator_state, strict=True, assign=True)
        selected_dtype = torch.float16 if selected_device.startswith("cuda") else torch.float32
        self.model = model.to(device=selected_device, dtype=selected_dtype).eval()
        del state, aggregator_state
        self.torch = torch
        self.device = torch.device(selected_device)
        self.dtype = selected_dtype
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.layer = int(layer)
        self.image_size = int(image_size)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        import torch.nn.functional as functional

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        original_h, original_w = value.shape[:2]
        max_dim = max(original_h, original_w)
        pad_left = (max_dim - original_w) // 2
        pad_top = (max_dim - original_h) // 2
        padded = np.zeros((max_dim, max_dim, 3), dtype=value.dtype)
        padded[
            pad_top : pad_top + original_h,
            pad_left : pad_left + original_w,
        ] = value
        tensor = self.torch.from_numpy(padded).permute(2, 0, 1)[None, None].to(
            device=self.device, dtype=self.torch.float32
        )
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        tensor = functional.interpolate(
            tensor.flatten(0, 1),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).unflatten(0, (1, 1)).to(dtype=self.dtype)
        autocast_enabled = self.device.type == "cuda"
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16,
            enabled=autocast_enabled,
        ):
            outputs, patch_start_index = self.model(tensor)
        tokens = outputs[self.layer]
        if tokens is None:
            raise RuntimeError(f"VGGT layer {self.layer} was not cached by the aggregator")
        tokens = tokens[0, 0, patch_start_index:, :].float().cpu().numpy()
        feature_size = self.image_size // self._PATCH_SIZE
        if tokens.shape[0] != feature_size * feature_size:
            raise RuntimeError(
                f"Unexpected VGGT patch count {tokens.shape[0]}; expected {feature_size ** 2}"
            )

        grid_y, grid_x = np.meshgrid(
            (np.arange(feature_size, dtype=np.float64) + 0.5) * self._PATCH_SIZE,
            (np.arange(feature_size, dtype=np.float64) + 0.5) * self._PATCH_SIZE,
            indexing="ij",
        )
        scale = self.image_size / float(max_dim)
        x1 = pad_left * scale
        y1 = pad_top * scale
        x2 = (pad_left + original_w) * scale
        y2 = (pad_top + original_h) * scale
        keep = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
        coordinates = np.stack(
            [
                (grid_x[keep] - x1) / scale,
                (grid_y[keep] - y1) / scale,
            ],
            axis=1,
        )
        descriptors = tokens.reshape(feature_size, feature_size, -1)[keep].astype(
            np.float64
        )
        scores = np.linalg.norm(descriptors, axis=1)
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": "VGGT-1B",
                "implementation": "https://github.com/facebookresearch/vggt.git",
                "implementation_commit": self.repository_commit,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": (
                    f"aggregator layer {self.layer} concatenated frame/global patch tokens"
                ),
                "preprocessing": (
                    f"RGB float [0,1]; black center pad to square; bicubic resize "
                    f"{self.image_size}x{self.image_size}; repository internal mean/std; "
                    f"{str(self.dtype).replace('torch.', '')}"
                ),
                "coordinate_system": (
                    "unpadded patch centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


class CroCoEncoderAdapter:
    """Pinned MASt3R/DUSt3R single-image encoder-token exporter."""

    _VALID_MODELS = {"mast3r", "dust3r"}

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        model_type: str,
        layer: int = 23,
        image_size: int = 512,
        device: str | None = None,
    ) -> None:
        normalized_type = model_type.strip().lower()
        if normalized_type not in self._VALID_MODELS:
            raise ValueError(
                f"model_type must be one of {sorted(self._VALID_MODELS)}"
            )
        if int(layer) < 0:
            raise ValueError("CroCo encoder layer must be non-negative")
        if image_size <= 0:
            raise ValueError("CroCo encoder image_size must be positive")
        repository_path = Path(repository).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if normalized_type == "mast3r":
            source = repository_path / "mast3r" / "model.py"
        else:
            source = repository_path / "dust3r" / "dust3r" / "model.py"
        if not source.is_file():
            raise FileNotFoundError(
                f"{normalized_type.upper()} source not found under {repository_path}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("CroCoEncoderAdapter requires torch") from error

        for path in (repository_path, repository_path / "dust3r"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        if normalized_type == "mast3r":
            from mast3r.model import load_model
        else:
            from dust3r.model import load_model

        # The official checkpoints store their constructor declaration under
        # ``args``; the pinned upstream loader is required to reconstruct the
        # exact architecture before loading its state dictionary.
        model = load_model(str(checkpoint_path), device="cpu", verbose=False).eval()
        block_count = len(model.enc_blocks)
        if int(layer) >= block_count:
            raise ValueError(
                f"CroCo encoder layer must be in [0, {block_count - 1}]"
            )
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.patch_embed.to(selected_device).eval()
        model.enc_blocks.to(selected_device).eval()
        model.enc_norm.to(selected_device).eval()
        self.model = model
        self.torch = torch
        self.device = torch.device(selected_device)
        self.repository = repository_path
        self.repository_commit = _repository_revision(repository_path)
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.model_type = normalized_type
        self.layer = int(layer)
        self.block_count = int(block_count)
        self.image_size = int(image_size)
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(set(int(value) for value in patch_size)) != 1:
                raise ValueError(f"Non-square CroCo patch size is unsupported: {patch_size}")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[Any, tuple[int, int], tuple[float, float, float, float]]:
        from PIL import Image

        value = np.asarray(image)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("image must be H x W or H x W x C")
        value = np.array(value[..., :3], copy=True, order="C")
        original_h, original_w = value.shape[:2]
        pil_image = Image.fromarray(value.astype(np.uint8), mode="RGB")
        scale = self.image_size / float(max(original_w, original_h))
        resized_w = int(round(original_w * scale))
        resized_h = int(round(original_h * scale))
        resampling = (
            Image.Resampling.LANCZOS
            if max(original_w, original_h) > self.image_size
            else Image.Resampling.BICUBIC
        )
        pil_image = pil_image.resize((resized_w, resized_h), resampling)
        centre_x, centre_y = resized_w // 2, resized_h // 2
        half_w = ((2 * centre_x) // 16) * 8
        half_h = ((2 * centre_y) // 16) * 8
        if resized_w == resized_h:
            half_h = int(3 * half_w / 4)
        left = float(centre_x - half_w)
        top = float(centre_y - half_h)
        right = float(centre_x + half_w)
        bottom = float(centre_y + half_h)
        pil_image = pil_image.crop((left, top, right, bottom))
        cropped = np.asarray(pil_image, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(np.array(cropped, copy=True, order="C"))
        tensor = tensor.permute(2, 0, 1)[None]
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device=self.device, dtype=self.torch.float32)
        return tensor, (original_h, original_w), (left, top, resized_w, resized_h)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        tensor, original_hw, resize_crop = self._preprocess(image)
        true_shape = self.torch.tensor(
            [[tensor.shape[-2], tensor.shape[-1]]],
            device=self.device,
            dtype=self.torch.int64,
        )
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16,
            enabled=self.device.type == "cuda",
        ):
            tokens, positions = self.model.patch_embed(tensor, true_shape=true_shape)
            for index, block in enumerate(self.model.enc_blocks):
                tokens = block(tokens, positions)
                if index == self.layer:
                    break
            if self.layer == self.block_count - 1:
                tokens = self.model.enc_norm(tokens)
        descriptors = tokens[0].float().cpu().numpy().astype(np.float64)
        cropped_h, cropped_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        feature_h = cropped_h // self.patch_size
        feature_w = cropped_w // self.patch_size
        if descriptors.shape[0] != feature_h * feature_w:
            raise RuntimeError(
                f"Unexpected CroCo token count {descriptors.shape[0]}; "
                f"expected {feature_h * feature_w}"
            )
        original_h, original_w = original_hw
        left, top, resized_w, resized_h = resize_crop
        scale_x = resized_w / float(original_w)
        scale_y = resized_h / float(original_h)
        grid_y, grid_x = np.meshgrid(
            (np.arange(feature_h, dtype=np.float64) + 0.5) * self.patch_size,
            (np.arange(feature_w, dtype=np.float64) + 0.5) * self.patch_size,
            indexing="ij",
        )
        coordinates = np.stack(
            [
                (grid_x.reshape(-1) + left) / scale_x,
                (grid_y.reshape(-1) + top) / scale_y,
            ],
            axis=1,
        )
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0.0, original_w - 1.0)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, original_h - 1.0)
        scores = np.linalg.norm(descriptors, axis=1)
        label = "MASt3R" if self.model_type == "mast3r" else "DUSt3R"
        features = LocalFeatures(
            coordinates_xy=coordinates,
            descriptors=descriptors,
            scores=scores,
            metadata={
                "model": label,
                "implementation": (
                    "https://github.com/naver/mast3r.git"
                    if self.model_type == "mast3r"
                    else "https://github.com/naver/dust3r.git"
                ),
                "implementation_commit": self.repository_commit,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256,
                "layer": (
                    f"single-image encoder block {self.layer}"
                    + (" with official final encoder normalization" if self.layer == self.block_count - 1 else "")
                ),
                "preprocessing": (
                    f"RGB normalized to [-1,1]; official long-edge resize {self.image_size}; "
                    f"center crop to multiples of 16; patch size {self.patch_size}; "
                    "repository autocast policy"
                ),
                "coordinate_system": (
                    "cropped patch centres mapped to zero-based original-image x,y pixels"
                ),
            },
        )
        features.validate()
        validate_model_metadata(features.metadata)
        return features


MODEL_EXPORT_CONTRACT: dict[str, dict[str, str]] = {
    "SIFT": {
        "mode": "built-in optional adapter",
        "representation": "128-D descriptor at detected keypoints",
    },
    "SuperPoint": {
        "mode": "built-in pinned LightGlue-repository adapter",
        "representation": "official/repository descriptor before matcher",
    },
    "XFeat": {
        "mode": "built-in pinned repository adapter",
        "representation": "official dense or sparse descriptor output",
    },
    "DeDoDe": {
        "mode": "built-in pinned repository adapter",
        "representation": "official descriptor output at selected keypoints",
    },
    "LoFTR": {
        "mode": "built-in pinned Kornia adapter",
        "representation": "declared coarse/fine feature tensor before matching",
    },
    "RoMa": {
        "mode": "built-in pinned repository adapter",
        "representation": "declared encoder feature-pyramid tensor before matching",
    },
    "DINOv2": {
        "mode": "built-in pinned repository adapter",
        "representation": "patch tokens from a pinned layer and checkpoint",
    },
    "VGGT": {
        "mode": "built-in pinned repository adapter",
        "representation": "cached frame/global aggregator patch tokens from a declared layer",
    },
    "MASt3R": {
        "mode": "built-in pinned repository adapter",
        "representation": "declared single-image CroCo encoder block tokens",
    },
    "DUSt3R": {
        "mode": "built-in pinned repository adapter (auxiliary)",
        "representation": "declared single-image CroCo encoder block tokens",
    },
    "ResNet-50": {
        "mode": "built-in pinned torchvision adapter",
        "representation": "pinned torchvision block output",
    },
}


def validate_model_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "model",
        "implementation",
        "implementation_commit",
        "checkpoint",
        "checkpoint_sha256",
        "layer",
        "preprocessing",
        "coordinate_system",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Model export metadata is missing fields: {missing}")
    blank = sorted(key for key in required if not str(metadata[key]).strip())
    if blank:
        raise ValueError(f"Model export metadata contains blank fields: {blank}")
