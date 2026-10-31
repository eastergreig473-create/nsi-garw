from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, sobel, zoom


EPS = 1e-12


def _float_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim not in {2, 3}:
        raise ValueError(f"Expected HxW or HxWxC image, received {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        limit = float(np.iinfo(value.dtype).max)
        value = value.astype(np.float64) / limit
    else:
        value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Image contains NaN or infinite values")
    return value


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    value = _float_image(image)
    if value.ndim == 2:
        return value
    if value.shape[2] == 1:
        return value[..., 0]
    if value.shape[2] < 3:
        raise ValueError("Color images require at least three channels")
    return (
        0.2126 * value[..., 0]
        + 0.7152 * value[..., 1]
        + 0.0722 * value[..., 2]
    )


def sobel_magnitude(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    gray = rgb_to_gray(image)
    horizontal = sobel(gray, axis=1, mode="reflect")
    vertical = sobel(gray, axis=0, mode="reflect")
    magnitude = np.hypot(horizontal, vertical)
    if normalize:
        scale = float(np.percentile(magnitude, 99.0))
        if scale > EPS:
            magnitude = np.clip(magnitude / scale, 0.0, 1.0)
    return magnitude


def depth_to_normals(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a depth map to unit normals and a validity mask.

    When intrinsics=(fx, fy, cx, cy) is supplied, normals are computed from
    neighboring 3-D points. Otherwise an orthographic depth-gradient proxy is
    used. Invalid depth pixels receive the zero vector and are excluded by mask.
    """
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    valid = np.isfinite(z) & (z > 0.0)
    # Finite differences must not treat missing laser samples as a zero-depth
    # surface. Require every pixel used by the gradient stencil to be valid.
    stencil_valid = valid.copy()
    stencil_valid[1:, :] &= valid[:-1, :]
    stencil_valid[:-1, :] &= valid[1:, :]
    stencil_valid[:, 1:] &= valid[:, :-1]
    stencil_valid[:, :-1] &= valid[:, 1:]
    safe = np.where(valid, z, 0.0)
    if intrinsics is None:
        dz_dy, dz_dx = np.gradient(safe)
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(safe)], axis=-1)
    else:
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("Focal lengths must be positive")
        rows, columns = np.indices(z.shape, dtype=np.float64)
        points = np.stack(
            [
                (columns - cx) * safe / fx,
                (rows - cy) * safe / fy,
                safe,
            ],
            axis=-1,
        )
        derivative_y = np.gradient(points, axis=0)
        derivative_x = np.gradient(points, axis=1)
        normals = np.cross(derivative_x, derivative_y)
        # Choose a consistent sign: non-negative camera-frame z component.
        normals = np.where((normals[..., 2:3] < 0.0), -normals, normals)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    stable = stencil_valid & (lengths[..., 0] > EPS)
    normals = np.divide(
        normals,
        lengths,
        out=np.zeros_like(normals),
        where=lengths > EPS,
    )
    normals[~stable] = 0.0
    return normals, stable


def structure_tensor(
    image: np.ndarray, sigma_derivative: float = 1.0, sigma_integration: float = 1.5
) -> np.ndarray:
    gray = rgb_to_gray(image)
    smoothed = gaussian_filter(gray, sigma=float(sigma_derivative), mode="reflect")
    gradient_y, gradient_x = np.gradient(smoothed)
    j_xx = gaussian_filter(
        gradient_x * gradient_x, sigma=float(sigma_integration), mode="reflect"
    )
    j_xy = gaussian_filter(
        gradient_x * gradient_y, sigma=float(sigma_integration), mode="reflect"
    )
    j_yy = gaussian_filter(
        gradient_y * gradient_y, sigma=float(sigma_integration), mode="reflect"
    )
    return np.stack([j_xx, j_xy, j_yy], axis=-1)


def _homography_jacobian(
    homography: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(homography, dtype=np.float64)
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    numerator_x = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    numerator_y = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    denominator_squared = denominator**2
    a11 = (h[0, 0] * denominator - numerator_x * h[2, 0]) / denominator_squared
    a12 = (h[0, 1] * denominator - numerator_x * h[2, 1]) / denominator_squared
    a21 = (h[1, 0] * denominator - numerator_y * h[2, 0]) / denominator_squared
    a22 = (h[1, 1] * denominator - numerator_y * h[2, 1]) / denominator_squared
    return a11, a12, a21, a22


def warp_structure_tensor(
    source_tensor: np.ndarray,
    homography_source_to_destination: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer a structure tensor through a homography.

    The source tensor is inverse-warped and transformed as
    J_destination=A^{-T} J_source A^{-1}, where A is the local Jacobian of
    the source-to-destination homography. The returned mask identifies valid,
    numerically stable destination pixels.
    """
    tensor = np.asarray(source_tensor, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError("source_tensor must have shape HxWx3 (Jxx, Jxy, Jyy)")
    homography = np.asarray(homography_source_to_destination, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape 3x3")
    inverse = np.linalg.inv(homography)
    rows, columns = np.indices(output_shape, dtype=np.float64)
    destination = np.stack(
        [columns.reshape(-1), rows.reshape(-1), np.ones(rows.size)], axis=0
    )
    source_homogeneous = inverse @ destination
    denominator = source_homogeneous[2]
    stable_h = np.abs(denominator) > EPS
    source_x = np.divide(
        source_homogeneous[0], denominator, out=np.zeros_like(denominator), where=stable_h
    )
    source_y = np.divide(
        source_homogeneous[1], denominator, out=np.zeros_like(denominator), where=stable_h
    )
    inside = (
        stable_h
        & (source_x >= 0.0)
        & (source_y >= 0.0)
        & (source_x <= tensor.shape[1] - 1)
        & (source_y <= tensor.shape[0] - 1)
    )
    sampled = np.stack(
        [
            map_coordinates(
                tensor[..., channel],
                [source_y, source_x],
                order=1,
                mode="constant",
                cval=0.0,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    a11, a12, a21, a22 = _homography_jacobian(
        homography, source_x, source_y
    )
    determinant = a11 * a22 - a12 * a21
    stable_jacobian = np.abs(determinant) > EPS
    inv11 = np.divide(a22, determinant, out=np.zeros_like(a22), where=stable_jacobian)
    inv12 = np.divide(-a12, determinant, out=np.zeros_like(a12), where=stable_jacobian)
    inv21 = np.divide(-a21, determinant, out=np.zeros_like(a21), where=stable_jacobian)
    inv22 = np.divide(a11, determinant, out=np.zeros_like(a11), where=stable_jacobian)
    j11, j12, j22 = sampled[:, 0], sampled[:, 1], sampled[:, 2]
    # A^{-T} J A^{-1}, expanded to avoid a large per-pixel matrix allocation.
    t11 = j11 * inv11 + j12 * inv21
    t12 = j11 * inv12 + j12 * inv22
    t21 = j12 * inv11 + j22 * inv21
    t22 = j12 * inv12 + j22 * inv22
    out11 = inv11 * t11 + inv21 * t21
    out12 = inv11 * t12 + inv21 * t22
    out22 = inv12 * t12 + inv22 * t22
    warped = np.stack([out11, out12, out22], axis=-1)
    valid = inside & stable_jacobian & np.isfinite(warped).all(axis=1)
    warped[~valid] = 0.0
    return warped.reshape(*output_shape, 3), valid.reshape(output_shape)


def tensor_cosine_proxy(
    first: np.ndarray, second: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.shape[-1] != 3:
        raise ValueError("Structure tensors must be aligned HxWx3 arrays")
    weight = np.sqrt(2.0)
    first_vector = np.stack(
        [first_array[..., 0], weight * first_array[..., 1], first_array[..., 2]],
        axis=-1,
    )
    second_vector = np.stack(
        [second_array[..., 0], weight * second_array[..., 1], second_array[..., 2]],
        axis=-1,
    )
    denominator = np.linalg.norm(first_vector, axis=-1) * np.linalg.norm(
        second_vector, axis=-1
    )
    cosine = np.divide(
        np.sum(first_vector * second_vector, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        # The denominator is a product of two norms, so compare against EPS^2.
        where=denominator > EPS**2,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != cosine.shape:
            raise ValueError("valid_mask is not aligned with the tensors")
        cosine = np.where(mask, cosine, np.nan)
    return cosine


def tensor_ncc_proxy(
    first: np.ndarray,
    second: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    sigma: float = 2.0,
) -> np.ndarray:
    """Local normalized cross-correlation of aligned structure-tensor fields."""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.ndim != 3 or first_array.shape[2] != 3:
        raise ValueError("Structure tensors must be aligned HxWx3 arrays")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    weight = np.sqrt(2.0)
    first_vector = first_array.copy()
    second_vector = second_array.copy()
    first_vector[..., 1] *= weight
    second_vector[..., 1] *= weight
    spatial_sigma = (float(sigma), float(sigma), 0.0)
    first_mean = gaussian_filter(first_vector, sigma=spatial_sigma, mode="reflect")
    second_mean = gaussian_filter(second_vector, sigma=spatial_sigma, mode="reflect")
    first_centered = first_vector - first_mean
    second_centered = second_vector - second_mean
    covariance = gaussian_filter(
        np.sum(first_centered * second_centered, axis=-1),
        sigma=float(sigma),
        mode="reflect",
    )
    first_variance = gaussian_filter(
        np.sum(first_centered**2, axis=-1), sigma=float(sigma), mode="reflect"
    )
    second_variance = gaussian_filter(
        np.sum(second_centered**2, axis=-1), sigma=float(sigma), mode="reflect"
    )
    denominator = np.sqrt(np.maximum(first_variance * second_variance, 0.0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > EPS**2,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != correlation.shape:
            raise ValueError("valid_mask is not aligned with the tensors")
        correlation = np.where(mask, correlation, np.nan)
    return correlation


def sample_map_at_coordinates(
    attribute_map: np.ndarray,
    coordinates_xy: np.ndarray,
    *,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(attribute_map, dtype=np.float64)
    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    if value.ndim not in {2, 3}:
        raise ValueError("attribute_map must be HxW or HxWxC")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_xy must have shape N x 2")
    x, y = coordinates[:, 0], coordinates[:, 1]
    valid = (
        np.isfinite(coordinates).all(axis=1)
        & (x >= 0.0)
        & (y >= 0.0)
        & (x <= value.shape[1] - 1)
        & (y <= value.shape[0] - 1)
    )
    if value.ndim == 2:
        sampled = map_coordinates(value, [y, x], order=order, mode="constant", cval=np.nan)[:, None]
    else:
        sampled = np.stack(
            [
                map_coordinates(value[..., channel], [y, x], order=order, mode="constant", cval=np.nan)
                for channel in range(value.shape[2])
            ],
            axis=1,
        )
    valid &= np.isfinite(sampled).all(axis=1)
    return sampled, valid


def resize_map(array: np.ndarray, output_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim not in {2, 3}:
        raise ValueError("Only two- or three-dimensional maps can be resized")
    factors = [output_shape[0] / value.shape[0], output_shape[1] / value.shape[1]]
    if value.ndim == 3:
        factors.append(1.0)
    resized = zoom(value, factors, order=int(order), mode="nearest", prefilter=order > 1)
    return resized[: output_shape[0], : output_shape[1], ...]
