from __future__ import annotations

from io import BytesIO

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter, rotate


def _image_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim not in {2, 3}:
        raise ValueError(f"Expected HxW or HxWxC image, received {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float64) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Image contains NaN or infinite values")
    return np.clip(value, 0.0, 1.0)


def _strength(value: float) -> float:
    strength = float(value)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    return strength


def texture_suppression(image: np.ndarray, strength: float) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    if level == 0.0:
        return value.copy()
    sigma = 0.25 + 5.75 * level
    sigma_value = (sigma, sigma, 0.0) if value.ndim == 3 else sigma
    blurred = gaussian_filter(value, sigma=sigma_value, mode="reflect")
    return np.clip((1.0 - 0.35 * level) * blurred + 0.35 * level * value.mean(), 0.0, 1.0)


def illumination_shift(image: np.ndarray, strength: float) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    gamma = float(np.exp(np.log(2.5) * level))
    adjusted = np.power(value, gamma)
    if adjusted.ndim == 3 and adjusted.shape[2] >= 3 and level > 0.0:
        rgb = adjusted[..., :3]
        hsv = rgb_to_hsv(rgb)
        hsv[..., 0] = np.mod(hsv[..., 0] + 0.18 * level, 1.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + 0.35 * level), 0.0, 1.0)
        adjusted = adjusted.copy()
        adjusted[..., :3] = hsv_to_rgb(hsv)
    return np.clip(adjusted, 0.0, 1.0)


def motion_blur(
    image: np.ndarray, strength: float, *, angle_degrees: float = 25.0
) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    if level == 0.0:
        return value.copy()
    length = max(3, int(round(3 + 22 * level)))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float64)
    kernel[length // 2, :] = 1.0
    kernel = rotate(kernel, float(angle_degrees), reshape=False, order=1, mode="constant")
    kernel = np.maximum(kernel, 0.0)
    kernel /= kernel.sum()
    if value.ndim == 2:
        return np.clip(convolve(value, kernel, mode="reflect"), 0.0, 1.0)
    channels = [convolve(value[..., index], kernel, mode="reflect") for index in range(value.shape[2])]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def jpeg_compression(image: np.ndarray, strength: float) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    if level == 0.0:
        return value.copy()
    quality = int(round(100 - 92 * level))
    encoded = np.rint(value * 255.0).astype(np.uint8)
    trailing: np.ndarray | None = None
    restore_channel_axis = False
    if encoded.ndim == 2:
        jpeg_input = encoded
        mode = "L"
    elif encoded.shape[2] == 1:
        jpeg_input = encoded[..., 0]
        mode = "L"
        restore_channel_axis = True
    elif encoded.shape[2] == 2:
        # JPEG has no two-channel mode. Compress the intensity channel and
        # preserve the auxiliary channel so the operator remains shape-safe.
        jpeg_input = encoded[..., 0]
        trailing = value[..., 1:]
        mode = "L"
        restore_channel_axis = True
    else:
        jpeg_input = encoded[..., :3]
        trailing = value[..., 3:] if encoded.shape[2] > 3 else None
        mode = "RGB"
    buffer = BytesIO()
    Image.fromarray(jpeg_input, mode=mode).save(
        buffer, format="JPEG", quality=quality, subsampling=0, optimize=False
    )
    buffer.seek(0)
    restored = np.asarray(Image.open(buffer), dtype=np.float64) / 255.0
    if restore_channel_axis:
        restored = restored[..., None]
    if trailing is not None:
        restored = np.concatenate([restored, trailing], axis=-1)
    return np.clip(restored, 0.0, 1.0)


def repetitive_pattern(
    image: np.ndarray, strength: float, *, seed: int = 0
) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    if level == 0.0:
        return value.copy()
    rng = np.random.default_rng(seed)
    height, width = value.shape[:2]
    patch_height = max(4, min(height, int(round(height * (0.08 + 0.12 * level)))))
    patch_width = max(4, min(width, int(round(width * (0.08 + 0.12 * level)))))
    top = int(rng.integers(0, max(1, height - patch_height + 1)))
    left = int(rng.integers(0, max(1, width - patch_width + 1)))
    patch = value[top : top + patch_height, left : left + patch_width]
    repetitions_y = int(np.ceil(height / patch_height))
    repetitions_x = int(np.ceil(width / patch_width))
    repeats = (repetitions_y, repetitions_x) + ((1,) if value.ndim == 3 else ())
    tiled = np.tile(patch, repeats)[:height, :width]
    alpha = 0.15 + 0.70 * level
    return np.clip((1.0 - alpha) * value + alpha * tiled, 0.0, 1.0)


def occlusion(image: np.ndarray, strength: float, *, seed: int = 0) -> np.ndarray:
    value = _image_float(image)
    level = _strength(strength)
    if level == 0.0:
        return value.copy()
    rng = np.random.default_rng(seed)
    height, width = value.shape[:2]
    target_area = height * width * (0.05 + 0.45 * level)
    aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    rectangle_height = min(height, max(1, int(round(np.sqrt(target_area / aspect)))))
    rectangle_width = min(width, max(1, int(round(np.sqrt(target_area * aspect)))))
    top = int(rng.integers(0, max(1, height - rectangle_height + 1)))
    left = int(rng.integers(0, max(1, width - rectangle_width + 1)))
    fill_shape = (1,) if value.ndim == 2 else (1, 1, value.shape[2])
    fill = rng.uniform(0.0, 1.0, size=fill_shape)
    output = value.copy()
    output[top : top + rectangle_height, left : left + rectangle_width] = fill
    return output


def composite_degradation(image: np.ndarray, strength: float, *, seed: int = 0) -> np.ndarray:
    level = _strength(strength)
    value = texture_suppression(image, 0.65 * level)
    value = illumination_shift(value, 0.75 * level)
    value = motion_blur(value, 0.55 * level, angle_degrees=25.0 + 20.0 * level)
    value = jpeg_compression(value, 0.70 * level)
    if level > 0.0:
        value = repetitive_pattern(value, 0.35 * level, seed=seed)
    return value


DEGRADATIONS = {
    "texture": texture_suppression,
    "illumination": illumination_shift,
    "motion_blur": motion_blur,
    "jpeg": jpeg_compression,
    "repetitive": repetitive_pattern,
    "occlusion": occlusion,
    "composite": composite_degradation,
}


def apply_degradation(
    image: np.ndarray, kind: str, strength: float, *, seed: int = 0
) -> np.ndarray:
    normalized = kind.strip().lower()
    if normalized not in DEGRADATIONS:
        raise ValueError(
            f"Unknown degradation {kind!r}; choose from {sorted(DEGRADATIONS)}"
        )
    function = DEGRADATIONS[normalized]
    if normalized in {"repetitive", "occlusion", "composite"}:
        return function(image, strength, seed=seed)  # type: ignore[misc]
    return function(image, strength)  # type: ignore[misc]
