"""
color_transform.py
    Linear mapping
        - Brightness modification:      g(x,y) = f(x,y) + b
        - Contrast modification:        g(x,y) = a * f(x,y)
        - Brightness + contrast:        g(x,y) = a * f(x,y) + b
        - Range mapping [f1,f2] -> [g1,g2]

    Non-linear mapping
        - Logarithmic mapping:          g(x,y) = c * log(f(x,y))
        - Exponential mapping:          g(x,y) = e^{f(x,y)}

    PDF-based mapping
        - Histogram Equalization
        - Histogram Specification (Matching)
"""

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert any BGR/float image into uint8 grayscale (0-255)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _normalize_to_01(gray: np.ndarray) -> np.ndarray:
    """Map grayscale values from 0-255 to float32 range [0, 1]."""
    gray = _to_gray(gray).astype(np.float32)
    return gray / 255.0


def _from_01_to_uint8(x: np.ndarray) -> np.ndarray:
    """Map float32 values in [0, 1] back to uint8 0-255."""
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------
# Linear mapping
# ----------------------------------------------------------------------


def brightness_adjust(img: np.ndarray, b: float) -> np.ndarray:
    """
    Adjust brightness by adding an offset.

    Args:
        img: Input image, grayscale or BGR uint8.
        b: Offset added to every pixel (positive brightens, negative darkens).

    Returns:
        np.ndarray: Brightness- shifted image clipped to [0, 255].
    """
    img_f = img.astype(np.float32)
    g = img_f + b
    return np.clip(g, 0, 255).astype(np.uint8)


def contrast_adjust(img: np.ndarray, a: float) -> np.ndarray:
    """
    Scale contrast by multiplying pixel values with a constant.

    Args:
        img: Input image, grayscale or BGR uint8.
        a: Contrast scale (>1 increases, 0<a<1 decreases).

    Returns:
        np.ndarray: Contrast-scaled image clipped to [0, 255].
    """
    img_f = img.astype(np.float32)
    g = a * img_f
    return np.clip(g, 0, 255).astype(np.uint8)


def brightness_contrast_adjust(img: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply contrast and brightness in one pass.

    Args:
        img: Input image, grayscale or BGR uint8.
        a: Contrast scale.
        b: Brightness offset.

    Returns:
        np.ndarray: Adjusted image clipped to [0, 255].
    """
    img_f = img.astype(np.float32)
    g = a * img_f + b
    return np.clip(g, 0, 255).astype(np.uint8)


def range_linear_mapping(
    gray: np.ndarray, f1: int, f2: int, g1: int, g2: int
) -> np.ndarray:
    """
    Linearly map a gray range [f1, f2] into [g1, g2].

    Args:
        gray: Input grayscale image.
        f1/f2: Input intensity bounds (inclusive).
        g1/g2: Output intensity bounds.

    Returns:
        np.ndarray: Gray image with the specified range remapped.
    """
    gray = _to_gray(gray)
    f1, f2 = int(f1), int(f2)
    g1, g2 = float(g1), float(g2)

    g = gray.astype(np.float32)
    mask = (gray >= f1) & (gray <= f2)
    if f2 == f1:
        return gray  # tránh chia 0

    g[mask] = g1 + (gray[mask] - f1) * (g2 - g1) / (f2 - f1)
    return np.clip(g, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Non-linear mapping
# ----------------------------------------------------------------------


def log_mapping(gray: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Apply logarithmic tone mapping to enhance dark regions.

    Args:
        gray: Input grayscale image.
        c: Log gain factor applied after normalization.

    Returns:
        np.ndarray: Log-mapped grayscale image.
    """
    f = _normalize_to_01(gray)
    g = c * np.log1p(f)  # log(1 + f)
    g = g / g.max()  # chuẩn hóa 0..1
    return _from_01_to_uint8(g)


def exp_mapping(gray: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Apply exponential tone mapping to emphasize bright regions.

    Args:
        gray: Input grayscale image.
        c: Exponential gain factor.

    Returns:
        np.ndarray: Exponentially mapped grayscale image.
    """
    f = _normalize_to_01(gray)
    g = np.exp(c * f) - 1.0  # nằm trong [0, e^c - 1]
    g = g / g.max()
    return _from_01_to_uint8(g)


# ----------------------------------------------------------------------
# PDF-based mapping – Histogram Equalization
# ----------------------------------------------------------------------


def histogram_equalization(gray: np.ndarray, n_levels: int = 256) -> np.ndarray:
    """
    Perform manual histogram equalization.

    Args:
        gray: Input grayscale image.
        n_levels: Number of intensity levels (default 256).

    Returns:
        np.ndarray: Equalized grayscale image.
    """
    gray = _to_gray(gray)
    N, M = gray.shape
    n_pixels = N * M

    # Step 1+2: Histogram H
    H = np.bincount(gray.flatten(), minlength=n_levels).astype(np.float64)

    # Step 3: cumulative histogram T
    T = np.cumsum(H)

    # Step 4: lookup table
    T_norm = np.round((n_levels - 1) * T / n_pixels).astype(np.uint8)

    # Step 5: mapping
    equalized = T_norm[gray]
    return equalized


# ----------------------------------------------------------------------
# PDF-based mapping – Histogram Specification (Matching)
# ----------------------------------------------------------------------


def histogram_specification(
    source: np.ndarray, reference: np.ndarray, n_levels: int = 256
) -> np.ndarray:
    """
    Match the histogram of a source image to a reference image.

    Args:
        source: Input grayscale image to be remapped.
        reference: Target grayscale image whose histogram is desired.
        n_levels: Number of intensity levels (default 256).

    Returns:
        np.ndarray: Histogram-matched grayscale image.
    """
    src = _to_gray(source)
    ref = _to_gray(reference)

    # Histogram
    Hs = np.bincount(src.flatten(), minlength=n_levels).astype(np.float64)
    Hr = np.bincount(ref.flatten(), minlength=n_levels).astype(np.float64)

    # CDF (chuẩn hóa 0..1)
    cdf_s = np.cumsum(Hs)
    cdf_s /= cdf_s[-1]

    cdf_r = np.cumsum(Hr)
    cdf_r /= cdf_r[-1]

    # Tạo bảng mapping: với mỗi mức k của source
    # tìm j sao cho |cdf_s[k] - cdf_r[j]| nhỏ nhất
    mapping = np.zeros(n_levels, dtype=np.uint8)
    j = 0
    for k in range(n_levels):
        while j < n_levels - 1 and cdf_r[j] < cdf_s[k]:
            j += 1
        mapping[k] = j

    # Ánh xạ
    matched = mapping[src]
    return matched


# ----------------------------------------------------------------------
# Demo nhanh khi chạy trực tiếp
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import os

    INPUT_PATH = "assets/Lenna.jpg"
    OUT_DIR = "outputs/color_transform"
    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(INPUT_PATH)

    gray = _to_gray(img)

    # Linear mappings
    brighter = brightness_adjust(gray, b=40)
    higher_contrast = contrast_adjust(gray, a=1.5)
    bc = brightness_contrast_adjust(gray, a=1.2, b=20)
    mapped_range = range_linear_mapping(gray, f1=50, f2=180, g1=0, g2=255)

    # Non-linear mappings
    log_img = log_mapping(gray, c=1.0)
    exp_img = exp_mapping(gray, c=1.5)

    # Histogram equalization
    heq = histogram_equalization(gray)

    # Histogram specification: dùng chính ảnh equalized làm reference demo
    hspec = histogram_specification(gray, heq)

    cv2.imwrite(os.path.join(OUT_DIR, "gray.png"), gray)
    cv2.imwrite(os.path.join(OUT_DIR, "bright.png"), brighter)
    cv2.imwrite(os.path.join(OUT_DIR, "contrast.png"), higher_contrast)
    cv2.imwrite(os.path.join(OUT_DIR, "bright_contrast.png"), bc)
    cv2.imwrite(os.path.join(OUT_DIR, "range_mapped.png"), mapped_range)
    cv2.imwrite(os.path.join(OUT_DIR, "log.png"), log_img)
    cv2.imwrite(os.path.join(OUT_DIR, "exp.png"), exp_img)
    cv2.imwrite(os.path.join(OUT_DIR, "hist_eq.png"), heq)
    cv2.imwrite(os.path.join(OUT_DIR, "hist_spec.png"), hspec)

    print(f"Đã lưu kết quả demo vào thư mục: {OUT_DIR}")
