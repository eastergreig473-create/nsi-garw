from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from PIL import Image

from .attributes import (
    depth_to_normals,
    sobel_magnitude,
    structure_tensor,
    tensor_cosine_proxy,
    tensor_ncc_proxy,
    warp_structure_tensor,
)
from .io import sha256_file


def _load_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}")
        dimensions = handle.readline().decode("ascii").strip().split()
        while dimensions and dimensions[0].startswith("#"):
            dimensions = handle.readline().decode("ascii").strip().split()
        if len(dimensions) != 2:
            raise ValueError(f"Invalid PFM dimensions in {path}")
        width, height = (int(value) for value in dimensions)
        scale = float(handle.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0.0 else ">f4"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(handle, dtype=dtype, count=width * height * channels)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"Truncated PFM file {path}: expected {expected} values")
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float64)


def load_array(path: str | Path, *, npz_key: str | None = None) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if npz_key is None:
                if len(archive.files) != 1:
                    raise ValueError(f"{source} contains multiple arrays; specify a key")
                npz_key = archive.files[0]
            if npz_key not in archive.files:
                raise ValueError(f"Array key {npz_key!r} is absent from {source}")
            return np.asarray(archive[npz_key])
    if suffix == ".pfm":
        return _load_pfm(source)
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as error:
            raise RuntimeError("Reading HDF5 depth requires h5py") from error
        with h5py.File(source, "r") as archive:
            dataset_names: list[str] = []
            archive.visititems(
                lambda name, item: dataset_names.append(name)
                if isinstance(item, h5py.Dataset)
                else None
            )
            selected = npz_key
            if selected is None:
                if "depth" in dataset_names:
                    selected = "depth"
                elif len(dataset_names) == 1:
                    selected = dataset_names[0]
                else:
                    raise ValueError(
                        f"{source} contains multiple HDF5 datasets; specify --depth-key"
                    )
            if selected not in archive:
                raise ValueError(f"Dataset key {selected!r} is absent from {source}")
            return np.asarray(archive[selected])
    if suffix in {".txt", ".csv", ""}:
        try:
            return np.asarray(np.loadtxt(source), dtype=np.float64)
        except (ValueError, UnicodeDecodeError):
            if suffix:
                raise
    return np.asarray(Image.open(source))


def save_depth_attributes(
    image_path: str | Path,
    depth_path: str | Path,
    output_path: str | Path,
    *,
    intrinsics: tuple[float, float, float, float] | None = None,
    depth_npz_key: str | None = None,
) -> dict[str, Any]:
    image = load_array(image_path)
    depth = np.asarray(load_array(depth_path, npz_key=depth_npz_key), dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.shape != image.shape[:2]:
        raise ValueError("Depth and image must have the same spatial dimensions")
    normals, normal_valid = depth_to_normals(depth, intrinsics)
    appearance = sobel_magnitude(image)
    metadata = {
        "annotation_type": "depth_normals_and_sobel",
        "image_path": str(Path(image_path).resolve()),
        "image_sha256": sha256_file(image_path),
        "depth_path": str(Path(depth_path).resolve()),
        "depth_sha256": sha256_file(depth_path),
        "intrinsics_fx_fy_cx_cy": list(intrinsics) if intrinsics is not None else None,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        geometry=normals,
        geometry_valid=normal_valid,
        appearance=appearance[..., None],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def save_hpatches_attributes(
    reference_image_path: str | Path,
    target_image_path: str | Path,
    homography_path: str | Path,
    output_path: str | Path,
    *,
    proxy: str = "ncc",
) -> dict[str, Any]:
    reference = load_array(reference_image_path)
    target = load_array(target_image_path)
    homography = np.asarray(load_array(homography_path), dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError("Homography must be a 3x3 array")
    reference_tensor = structure_tensor(reference)
    target_tensor = structure_tensor(target)
    transferred, valid = warp_structure_tensor(
        reference_tensor, homography, target.shape[:2]
    )
    if proxy == "ncc":
        geometry = tensor_ncc_proxy(transferred, target_tensor, valid)
    elif proxy == "cosine":
        geometry = tensor_cosine_proxy(transferred, target_tensor, valid)
    else:
        raise ValueError("proxy must be 'ncc' or 'cosine'")
    appearance = sobel_magnitude(target)
    metadata = {
        "annotation_type": "hpatches_homography_structure_tensor",
        "geometry_proxy": proxy,
        "reference_image_path": str(Path(reference_image_path).resolve()),
        "reference_image_sha256": sha256_file(reference_image_path),
        "target_image_path": str(Path(target_image_path).resolve()),
        "target_image_sha256": sha256_file(target_image_path),
        "homography_path": str(Path(homography_path).resolve()),
        "homography_sha256": sha256_file(homography_path),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        geometry=geometry[..., None],
        geometry_valid=np.isfinite(geometry) & valid,
        appearance=appearance[..., None],
        transferred_structure_tensor=transferred,
        target_structure_tensor=target_tensor,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata
