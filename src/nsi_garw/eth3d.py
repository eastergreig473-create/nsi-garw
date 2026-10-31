"""ETH3D float32 depth and calibrated distortion handling."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def read_raw_depth(path: str | Path, width: int, height: int) -> np.ndarray:
    """The official .JPG-suffixed depth files are little-endian float dumps."""
    source = Path(path)
    if source.stat().st_size != width * height * 4:
        raise ValueError(f"Raw ETH3D depth size disagrees with camera dimensions: {source}")
    values = np.fromfile(source, dtype="<f4").reshape(height, width)
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def read_cameras(path: str | Path) -> dict:
    import pycolmap
    cameras = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        identifier, model, width, height, *parameters = line.split()
        cameras[int(identifier)] = pycolmap.Camera(camera_id=int(identifier), model=model,
            width=int(width), height=int(height), params=[float(p) for p in parameters])
    if not cameras:
        raise ValueError(f"No camera records in {path}")
    return cameras


def read_images(path: str | Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    images = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"Invalid COLMAP image record: {line}")
        identifier, qw, qx, qy, qz, tx, ty, tz, camera_id, name = fields
        images[Path(name).name] = {"image_id": int(identifier), "camera_id": int(camera_id),
            "quaternion_wxyz": [float(v) for v in (qw, qx, qy, qz)],
            "translation": [float(v) for v in (tx, ty, tz)], "name": name}
        # The following points2D record may be an empty line and must be consumed.
        if index >= len(lines):
            raise ValueError("Missing points2D line in COLMAP images.txt")
        index += 1
    return images


def depth_on_pinhole(raw_depth: np.ndarray, original_camera, pinhole_camera,
                     *, block_rows: int = 128) -> np.ndarray:
    """Inverse-map PINHOLE pixel centers into the original distorted depth.

    Use pycolmap's exact camera models. Nearest-neighbor sampling preserves
    missing laser measurements. Outside points remain invalid, never clipped
    onto an image border. Camera coordinates use the top-left pixel corner as
    origin; array index (0,0) therefore has center coordinates (0.5,0.5).
    """
    if pinhole_camera.model_name != "PINHOLE":
        raise ValueError("Target camera must be PINHOLE")
    if raw_depth.shape != (original_camera.height, original_camera.width):
        raise ValueError("Original depth shape does not match original camera")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    height, width = pinhole_camera.height, pinhole_camera.width
    output = np.full((height, width), np.nan, dtype=np.float32)
    for start in range(0, height, block_rows):
        end = min(height, start + block_rows)
        yy, xx = np.meshgrid(np.arange(start, end) + .5, np.arange(width) + .5, indexing="ij")
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        rays_xy = pinhole_camera.cam_from_img(xy)
        rays = np.column_stack([rays_xy, np.ones(len(xy))])
        mapped = original_camera.img_from_cam(rays)
        valid = (np.isfinite(mapped).all(axis=1) & (mapped[:, 0] >= 0)
            & (mapped[:, 0] < original_camera.width) & (mapped[:, 1] >= 0)
            & (mapped[:, 1] < original_camera.height))
        indices = np.floor(mapped[valid]).astype(np.int64)
        sampled = raw_depth[indices[:, 1], indices[:, 0]]
        finite = np.isfinite(sampled) & (sampled > 0)
        destination = np.flatnonzero(valid)[finite]
        block = output[start:end].reshape(-1)
        block[destination] = sampled[finite]
    return output
