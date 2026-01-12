"""
geometric_transforms.py
    1. Interpolation techniques
    2. Affine coordinate transforms
    3. Backward mapping engine
    4. Combined wrappers + demo helpers
"""

from __future__ import annotations

import os
import numpy as np
import cv2


# ==========================
# 0. Common utility
# ==========================


def _ensure_dir(path: str):
    """Create the folder if missing (used by demo writers)."""
    os.makedirs(path, exist_ok=True)


def _as_float32(image: np.ndarray) -> np.ndarray:
    """Return a float32 view of the image without copying when possible."""
    return image.astype(np.float32, copy=False)


def _alloc_like(image: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """
    Allocate a zero array with the same channel layout as the source image.

    Args:
        image: Input image to mimic.
        out_size: Desired height, width.

    Returns:
        np.ndarray: Zero array (float32) with proper shape.
    """
    h_out, w_out = out_size
    if image.ndim == 2:
        return np.zeros((h_out, w_out), dtype=np.float32)
    return np.zeros((h_out, w_out, image.shape[2]), dtype=np.float32)


def _save_image(out_dir: str, filename: str, image: np.ndarray) -> str:
    """
    Utility that writes an image and returns its absolute path.
    """
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, image)
    return path


# ==========================
# 1. INTERPOLATION TECHNIQUES
# ==========================


def interp_nearest(image: np.ndarray, x: float, y: float):
    """
    Sample an image using nearest-neighbor interpolation.

    Args:
        image: Input image (gray or color).
        x/y: Continuous coordinates in source space.

    Returns:
        np.ndarray | float: Pixel value (float32) or vector for color.
    """
    h, w = image.shape[:2]
    ix = int(round(x))
    iy = int(round(y))

    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        if image.ndim == 2:
            return 0.0
        return np.zeros(image.shape[2:], dtype=np.float32)

    return _as_float32(image[iy, ix])


def interp_bilinear(image: np.ndarray, x: float, y: float):
    """
    Sample an image using bilinear interpolation.

    Args:
        image: Input image (gray or color).
        x/y: Continuous coordinates in source space.

        Returns:
            np.ndarray | float: Interpolated value(s).
    """
    h, w = image.shape[:2]

    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return interp_nearest(image, x, y)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    a = x - x0
    b = y - y0

    img_f = _as_float32(image)

    if image.ndim == 2:
        f00 = img_f[y0, x0]
        f10 = img_f[y0, x1]
        f01 = img_f[y1, x0]
        f11 = img_f[y1, x1]
    else:
        f00 = img_f[y0, x0, :]
        f10 = img_f[y0, x1, :]
        f01 = img_f[y1, x0, :]
        f11 = img_f[y1, x1, :]

    value = (
        (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 + (1 - a) * b * f01 + a * b * f11
    )

    return value


# ==========================
# 2. AFFINE COORDINATE TRANSFORM
# ==========================


def affine_forward(A_2x3: np.ndarray, x: float, y: float):
    """
    Apply a forward affine transform to a coordinate.

    Args:
        A_2x3: Affine matrix.
        x/y: Source coordinate.

    Returns:
        Tuple[float, float]: Transformed coordinate.
    """
    vec = np.array([x, y, 1.0], dtype=np.float32)
    x_prime, y_prime = A_2x3 @ vec
    return x_prime, y_prime


def build_affine_homogeneous(A_2x3: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine matrix into homogeneous 3x3 form."""
    A_h = np.eye(3, dtype=np.float32)
    A_h[:2, :] = A_2x3
    return A_h


def affine_backward(A_inv_3x3: np.ndarray, x_prime: float, y_prime: float):
    """
    Apply the inverse affine matrix to a destination coordinate.

    Args:
        A_inv_3x3: Inverted homogeneous matrix.
        x_prime/y_prime: Destination coordinate.

    Returns:
        Tuple[float, float]: Source coordinate.
    """
    vec_p = np.array([x_prime, y_prime, 1.0], dtype=np.float32)
    x, y, _ = A_inv_3x3 @ vec_p
    return x, y


def forward_mapping(image: np.ndarray, A_2x3: np.ndarray, out_size: tuple[int, int]):
    """
    Forward-map an image for demonstration purposes (gaps allowed).

    Args:
        image: Input image.
        A_2x3: Affine matrix.
        out_size: Output height and width.

    Returns:
        np.ndarray: Forward-mapped float32 image.
    """
    h_out, w_out = out_size
    out = _alloc_like(image, out_size)
    h, w = image.shape[:2]

    for y in range(h):
        for x in range(w):
            x_prime, y_prime = affine_forward(A_2x3, x, y)
            xi = int(round(x_prime))
            yi = int(round(y_prime))
            if 0 <= xi < w_out and 0 <= yi < h_out:
                out[yi, xi] = image[y, x]

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================
# 3. BACKWARD MAPPING ENGINE
# ==========================


def backward_mapping(
    image: np.ndarray, A_2x3: np.ndarray, out_size: tuple[int, int], interpolate_func
):
    """
    Simplified backward mapping using a provided affine matrix.

    Args:
        image: Source image.
        A_2x3: Affine matrix (already inverse if needed).
        out_size: Output shape (h, w).
        interpolate_func: Callable used to sample source pixels.

    Returns:
        np.ndarray: Warp result as uint8.
    """
    h_out, w_out = out_size
    A_h = build_affine_homogeneous(A_2x3)
    A_inv = np.linalg.inv(A_h)

    g = _alloc_like(image, out_size)

    for y_prime in range(h_out):
        for x_prime in range(w_out):
            x, y = affine_backward(A_inv, x_prime, y_prime)
            value = interpolate_func(image, x, y)
            g[y_prime, x_prime] = value

    return np.clip(g, 0, 255).astype(np.uint8)


# ==========================
# 4. COMBINED WRAPPERS (TECHNIQUE + APPLICATION)
# ==========================


def warp_affine_nearest(
    image: np.ndarray, A_2x3: np.ndarray, out_size: tuple[int, int]
):
    """Warp helper that forces nearest-neighbor interpolation."""
    return backward_mapping(image, A_2x3, out_size, interpolate_func=interp_nearest)


def warp_affine_bilinear(
    image: np.ndarray, A_2x3: np.ndarray, out_size: tuple[int, int]
):
    """Warp helper that forces bilinear interpolation."""
    return backward_mapping(image, A_2x3, out_size, interpolate_func=interp_bilinear)


# ---- Build matrices cho từng ứng dụng ----


def build_scale_matrix(image: np.ndarray, sx=1.5, sy=1.5):
    """
    Construct a scaling matrix and output size for the given image.
    """
    h, w = image.shape[:2]
    A = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
    out_size = (int(round(h * sy)), int(round(w * sx)))
    return A, out_size


def build_rotation_matrix(image: np.ndarray, angle_deg=30):
    """
    Construct a rotation matrix about the image center.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    T_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)

    A_h = T_pos @ R @ T_neg
    return A_h[:2, :], (h, w)


def build_translation_matrix(image: np.ndarray, tx=40, ty=25):
    """Return a pure translation matrix for completeness."""
    h, w = image.shape[:2]
    A = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return A, (h, w)


# def build_shear_matrix(image: np.ndarray, kx=0.4, ky=0.0):
#     h, w = image.shape[:2]
#     A = np.array([[1, kx, 0], [ky, 1, 0]], dtype=np.float32)
#     return A, (h, w)


def build_shear_matrix(image: np.ndarray, kx=0.4, ky=0.0):
    """
    Construct a shear matrix and an output size large enough
    to contain the whole sheared image (giống cách cv2.warpAffine làm).

    x' = x + kx * y
    y' = ky * x + y
    """
    h, w = image.shape[:2]

    # Ma trận shear (forward transform, giống cv2)
    A = np.array([[1, kx, 0], [ky, 1, 0]], dtype=np.float32)

    # Trường hợp phổ biến: shear ngang (ky = 0) -> mở rộng chiều rộng giống cv2
    if ky == 0.0:
        out_w = w + int(abs(kx * h))
        out_h = h
        return A, (out_h, out_w)

    # Trường hợp shear dọc (kx = 0) -> mở rộng chiều cao tương tự
    if kx == 0.0:
        out_w = w
        out_h = h + int(abs(ky * w))
        return A, (out_h, out_w)

    # Trường hợp tổng quát: tính bounding box của 4 góc sau shear,
    # rồi dịch ma trận A để ảnh không bị lệch ra ngoài canvas.
    corners = np.array(
        [
            [0, 0, 1],
            [w - 1, 0, 1],
            [0, h - 1, 1],
            [w - 1, h - 1, 1],
        ],
        dtype=np.float32,
    ).T  # shape (3, 4)

    warped = A @ corners  # shape (2, 4)
    xs = warped[0]
    ys = warped[1]

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    out_w = int(np.ceil(max_x - min_x + 1))
    out_h = int(np.ceil(max_y - min_y + 1))

    # Dịch ma trận shear để toàn bộ ảnh nằm trong [0, out_w) x [0, out_h)
    A[0, 2] -= min_x
    A[1, 2] -= min_y

    return A, (out_h, out_w)


# ---- Demo helpers ----


def demo_scaling(image: np.ndarray, out_dir: str):
    """Write scaling demo images for both interpolation modes."""
    _ensure_dir(out_dir)
    A, out_size = build_scale_matrix(image)
    g_near = warp_affine_nearest(image, A, out_size)
    g_bili = warp_affine_bilinear(image, A, out_size)
    return [
        ("scale_nearest", _save_image(out_dir, "scale_nearest.png", g_near)),
        ("scale_bilinear", _save_image(out_dir, "scale_bilinear.png", g_bili)),
    ]


def demo_rotation(image: np.ndarray, out_dir: str):
    """Write rotation demo images for both interpolation modes."""
    _ensure_dir(out_dir)
    A, out_size = build_rotation_matrix(image)
    g_near = warp_affine_nearest(image, A, out_size)
    g_bili = warp_affine_bilinear(image, A, out_size)
    return [
        ("rotate_nearest", _save_image(out_dir, "rotate_nearest.png", g_near)),
        ("rotate_bilinear", _save_image(out_dir, "rotate_bilinear.png", g_bili)),
    ]


def demo_shear(image: np.ndarray, out_dir: str):
    """Write shear demo images for both interpolation modes."""
    _ensure_dir(out_dir)
    A, out_size = build_shear_matrix(image)
    g_near = warp_affine_nearest(image, A, out_size)
    g_bili = warp_affine_bilinear(image, A, out_size)
    return [
        ("shear_nearest", _save_image(out_dir, "shear_nearest.png", g_near)),
        ("shear_bilinear", _save_image(out_dir, "shear_bilinear.png", g_bili)),
    ]


def demo_interpolation(image: np.ndarray, out_dir: str):
    """Visualize interpolation-only effect using an upsample warp."""
    _ensure_dir(out_dir)
    A, out_size = build_scale_matrix(image, sx=1.6, sy=1.6)
    g_near = warp_affine_nearest(image, A, out_size)
    g_bili = warp_affine_bilinear(image, A, out_size)
    return [
        ("nearest_only", _save_image(out_dir, "nearest_only.png", g_near)),
        ("bilinear_only", _save_image(out_dir, "bilinear_only.png", g_bili)),
    ]


def demo_affine_coordinate(image: np.ndarray, out_dir: str):
    """Demonstrate forward mapping of coordinates only."""
    _ensure_dir(out_dir)
    A, out_size = build_rotation_matrix(image, angle_deg=20)
    forward_img = forward_mapping(image, A, out_size)
    return [
        (
            "affine_forward_mapping",
            _save_image(out_dir, "affine_forward.png", forward_img),
        ),
    ]


def demo_backward_engine(image: np.ndarray, out_dir: str):
    """Demonstrate backward mapping with both interpolators."""
    _ensure_dir(out_dir)
    A, out_size = build_rotation_matrix(image, angle_deg=20)
    g_near = backward_mapping(image, A, out_size, interpolate_func=interp_nearest)
    g_bili = backward_mapping(image, A, out_size, interpolate_func=interp_bilinear)
    return [
        ("backward_nearest", _save_image(out_dir, "backward_nearest.png", g_near)),
        ("backward_bilinear", _save_image(out_dir, "backward_bilinear.png", g_bili)),
    ]


def demo_combined(image: np.ndarray, out_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Generate application demos (scale/rotate/shear) and collect paths."""
    outputs: dict[str, list[tuple[str, str]]] = {}
    outputs["scale"] = demo_scaling(image, os.path.join(out_dir, "scale"))
    outputs["rotate"] = demo_rotation(image, os.path.join(out_dir, "rotation"))
    outputs["shear"] = demo_shear(image, os.path.join(out_dir, "shear"))
    return outputs


def run_geometric_experiments(image: np.ndarray, out_dir: str):
    """
    Run all geometric experiments and return references to output files.

    Args:
        image: Input grayscale or color image.
        out_dir: Directory to store intermediate PNGs.

    Returns:
        dict[str, list[tuple[str, str]]]: Mapping category -> list of outputs.
    """
    _ensure_dir(out_dir)
    results: dict[str, list[tuple[str, str]]] = {}

    results["interpolation"] = demo_interpolation(
        image, os.path.join(out_dir, "interpolation")
    )
    results["affine_transform"] = demo_affine_coordinate(
        image, os.path.join(out_dir, "affine_transform")
    )
    results["backward_mapping"] = demo_backward_engine(
        image, os.path.join(out_dir, "backward_mapping")
    )
    combined = demo_combined(image, os.path.join(out_dir, "combined"))
    for comp_name, outputs in combined.items():
        results[f"combined_{comp_name}"] = outputs

    return results


if __name__ == "__main__":
    OUTPUT_DIR = "outputs/geometric_transforms"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread("assets/Lenna.jpg", cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("assets/Lenna.jpg")
    run_geometric_experiments(img, OUTPUT_DIR)
