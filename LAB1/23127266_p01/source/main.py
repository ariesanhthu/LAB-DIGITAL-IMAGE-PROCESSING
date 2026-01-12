from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import urllib.request

from color_transform import (
    brightness_adjust,
    brightness_contrast_adjust,
    contrast_adjust,
    exp_mapping,
    histogram_equalization,
    histogram_specification,
    log_mapping,
    range_linear_mapping,
)
from geometric_transforms import (
    run_geometric_experiments,
    build_scale_matrix,
    build_rotation_matrix,
    build_translation_matrix,
    build_shear_matrix,
    warp_affine_nearest,
    warp_affine_bilinear,
)

LENA_URL = "http://www.ess.ic.kanagawa-it.ac.jp/std_img/colorimage/Lenna.jpg"
DEFAULT_LENA_PATH = Path("assets") / "Lenna.jpg"

GEOMETRIC_COMPONENTS = {
    "combined_scale": ("scale", "scale_compare.png"),
    "combined_rotate": ("rotate", "rotate_compare.png"),
    "combined_shear": ("shear", "shear_compare.png"),
}


# ---------------------------------------------------------------------------
# CLI parsing & IO helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Build the CLI parser and extract arguments.

    Returns:
        argparse.Namespace: Parsed values for mode, image path, and out_dir.
    """
    parser = argparse.ArgumentParser(
        description="Basic color & geometric transforms demo."
    )
    parser.add_argument(
        "--mode",
        choices=["test", "batch", "interactive"],
        default="test",
        help="test (default), batch, interactive.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Ảnh đầu vào (batch mode). Bỏ trống để chương trình hỏi.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs"),
        help="Thư mục lưu kết quả.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    """Create the directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """
    Load an image from disk in BGR format.

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: Loaded BGR image.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh tại {path}")
    return img


def fetch_lena_image(dest: Path = DEFAULT_LENA_PATH) -> np.ndarray:
    """
    Ensure a local copy of Lena exists and return it.

    Args:
        dest: Destination path for the cached Lena image.

    Returns:
        np.ndarray: Loaded BGR Lena image.
    """
    ensure_output_dir(dest.parent)
    if not dest.exists():
        print(f"[INFO] Tải Lena về {dest}")
        urllib.request.urlretrieve(LENA_URL, dest)
    else:
        print(f"[INFO] Dùng Lena có sẵn tại {dest}")
    return load_image(dest)


def choose_image_interactively() -> Tuple[np.ndarray, str]:
    """
    Prompt the user to pick Lena or provide a custom path.

    Returns:
        Tuple[np.ndarray, str]: Loaded image and a human-readable source label.
    """
    menu = (
        "\nChọn nguồn ảnh:\n"
        "[1] Lena chuẩn (download nếu thiếu)\n"
        "[2] Nhập đường dẫn ảnh bất kỳ\n"
    )
    print(menu)
    while True:
        choice = input("Lựa chọn [1/2] (Enter = 1): ").strip() or "1"
        if choice == "1":
            img = fetch_lena_image()
            return img, "Lena"
        if choice == "2":
            raw = input("Nhập đường dẫn ảnh: ").strip().strip('"').strip("'")
            if not raw:
                print("Đường dẫn trống, thử lại.")
                continue
            path = Path(raw)
            try:
                img = load_image(path)
            except FileNotFoundError as exc:
                print(exc)
                continue
            return img, str(path)
        print("Chỉ nhận '1' hoặc '2'. Thử lại.")


# ---------------------------------------------------------------------------
# Core processing helpers
# ---------------------------------------------------------------------------


def color_transforms(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run all predefined color/intensity transforms.

    Args:
        image: Input BGR image.

    Returns:
        Dict[str, np.ndarray]: Mapping from transform name to resulting image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq = histogram_equalization(gray)
    hist_spec = histogram_specification(gray, hist_eq)
    return {
        "bright_plus40": brightness_adjust(image, b=40),
        "contrast_x1.4": contrast_adjust(image, a=1.4),
        "bright_contrast": brightness_contrast_adjust(image, a=1.2, b=25),
        "range_map_40_200_to_0_255": range_linear_mapping(gray, 40, 200, 0, 255),
        "log_mapping": log_mapping(gray, c=1.0),
        "exp_mapping": exp_mapping(gray, c=1.2),
        "hist_equalization": hist_eq,
        "hist_specification": hist_spec,
    }


def save_results(results: Dict[str, np.ndarray], out_dir: Path) -> None:
    """
    Persist a dictionary of images to disk as PNG files.

    Args:
        results: Mapping from file stem to image.
        out_dir: Target directory.
    """
    ensure_output_dir(out_dir)
    for name, img in results.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), img)


# ---------------------------------------------------------------------------
# Compare sheet builders
# ---------------------------------------------------------------------------


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Ensure the image has three channels for visualization.

    Args:
        img: Grayscale or color image.

    Returns:
        np.ndarray: BGR image.
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def _make_tile(img: np.ndarray, label: str, size=(256, 256)) -> np.ndarray:
    """
    Resize an image and add a text header for comparison grids.

    Args:
        img: Image to display.
        label: Text shown above the tile.
        size: Desired tile size (width, height).

    Returns:
        np.ndarray: Labeled tile.
    """
    tile = cv2.resize(_to_bgr(img), size)
    header_h = 30
    header = np.zeros((header_h, tile.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (tile.shape[1], header_h), (35, 35, 35), -1)
    cv2.putText(
        header,
        label.upper()[:32],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, tile])


def save_color_compares(
    original: np.ndarray,
    gray_original: np.ndarray,
    color_results: Dict[str, np.ndarray],
    out_dir: Path,
    tile_size=(256, 256),
) -> None:
    """
    Build pairwise comparison images for color transforms.

    Args:
        original: Original BGR image.
        gray_original: Original image converted to grayscale.
        color_results: Outputs from color_transforms().
        out_dir: Output directory for compare images.
        tile_size: Tile size for visualization.
    """
    ensure_output_dir(out_dir)
    cv_refs: Dict[str, np.ndarray] = {
        "hist_equalization": cv2.equalizeHist(gray_original),
    }
    for name, img in color_results.items():
        cv_img = cv_refs.get(name)
        if cv_img is None:
            continue
        tiles = [
            _make_tile(cv_img, "cv2.equalizeHist", tile_size),
            _make_tile(img, name, tile_size),
        ]
        collage = cv2.hconcat(tiles)
        cv2.imwrite(str(out_dir / f"{name}_compare.png"), collage)


def _cv_geometric_baseline(category: str, src: np.ndarray | None):
    """
    Compute OpenCV warp baseline for a geometric category.

    Args:
        category: Combined category key (scale/rotate/shear).
        src: Source image (color or gray).

    Returns:
        np.ndarray | None: Baseline image or None if unsupported.
    """
    if src is None:
        return None

    if category == "combined_scale":
        A, out_size = build_scale_matrix(src)
        size = (out_size[1], out_size[0])
        return cv2.warpAffine(src, A, size)

    if category == "combined_rotate":
        A, out_size = build_rotation_matrix(src)
        size = (out_size[1], out_size[0])
        return cv2.warpAffine(src, A, size)

    if category == "combined_shear":
        h, w = src.shape[:2]
        kx = 0.4
        ky = 0.0
        M = np.array([[1, kx, 0], [ky, 1, 0]], dtype=np.float32)
        out_w = w + int(abs(kx * h))
        return cv2.warpAffine(src, M, (out_w, h))

    return None


def save_geometric_compares(
    original: np.ndarray,
    original_gray: np.ndarray,
    suite_outputs: Dict[str, list[tuple[str, str]]],
    out_dir: Path,
    tile_size=(256, 256),
) -> None:
    """
    Compose comparison sheets for scale/rotate/shear demos.

    Args:
        original: Original BGR image.
        original_gray: Grayscale version of the original.
        suite_outputs: Dict returned by run_geometric_experiments().
        out_dir: Output directory for comparison PNGs.
        tile_size: Tile size for visualization.
    """
    ensure_output_dir(out_dir)

    base_color = original if original.ndim == 3 else None
    base_gray = original_gray

    cv_baseline_color = {
        key: _cv_geometric_baseline(key, base_color) for key in GEOMETRIC_COMPONENTS
    }
    cv_baseline_gray = {
        key: _cv_geometric_baseline(key, base_gray) for key in GEOMETRIC_COMPONENTS
    }

    for key, (title, filename) in GEOMETRIC_COMPONENTS.items():
        outputs = suite_outputs.get(key)
        if not outputs:
            continue
        tiles = []
        for name, path_str in outputs:
            img = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if img is None:
                continue
            tiles.append(_make_tile(img, name, tile_size))
        if not tiles:
            continue
        # Determine baseline matching color/grayscale of outputs
        sample_path = outputs[0][1]
        sample_img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
        is_gray = sample_img is not None and sample_img.ndim == 2
        baseline = cv_baseline_gray.get(key) if is_gray else cv_baseline_color.get(key)
        if baseline is None:
            baseline = (
                base_gray
                if is_gray
                else (base_color if base_color is not None else base_gray)
            )
        label = "cv2 (gray)" if is_gray else "cv2 (rgb)"
        tiles.append(_make_tile(baseline, label, tile_size))
        collage = cv2.hconcat(tiles)
        header = np.zeros((40, collage.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(header, (0, 0), (collage.shape[1], 40), (60, 60, 60), -1)
        # Removed the row with task | thuật 1 | thuật 2 | cv as instructed
        block = cv2.vconcat([header, collage])
        cv2.imwrite(str(out_dir / filename), block)


# ---------------------------------------------------------------------------
# Full pipeline runner (used by test & batch modes)
# ---------------------------------------------------------------------------


def run_full_pipeline(image: np.ndarray, out_dir: Path, source_desc: str) -> None:
    """
    Execute color and geometric pipelines plus comparisons.

    Args:
        image: Input BGR image.
        out_dir: Root directory for outputs.
        source_desc: Text description of the source image.
    """
    ensure_output_dir(out_dir)

    save_results({"original": image}, out_dir)

    color_results = color_transforms(image)
    save_results(color_results, out_dir / "color")

    suite_outputs = run_geometric_experiments(
        image, out_dir / "geometric_transforms_demo"
    )

    compare_dir = out_dir / "compare"
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_color_compares(image, gray_original, color_results, compare_dir / "color")
    save_geometric_compares(
        image, gray_original, suite_outputs, compare_dir / "geometric"
    )

    print(f"Đã xử lý ảnh từ: {source_desc}")
    print(f"Tất cả kết quả nằm trong: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Modes: test / batch / interactive
# ---------------------------------------------------------------------------


def run_mode_test(out_dir: Path) -> None:
    """Run the full pipeline using the canonical Lena image."""
    image = fetch_lena_image()
    run_full_pipeline(image, out_dir, "Lena (test mode)")


def run_mode_batch(image_path: Path | None, out_dir: Path) -> None:
    """
    Run the pipeline using a user-provided image or interactive selection.

    Args:
        image_path: Optional CLI path.
        out_dir: Output directory.
    """
    if image_path:
        image = load_image(image_path)
        source = str(image_path)
    else:
        image, source = choose_image_interactively()
    run_full_pipeline(image, out_dir, source)


def _input_float(prompt: str, default: float) -> float:
    """Read a float from stdin with a default fallback."""
    raw = input(f"{prompt} (default={default}): ").strip()
    return float(raw) if raw else default


def _input_int(prompt: str, default: int) -> int:
    """Read an int from stdin with a default fallback."""
    raw = input(f"{prompt} (default={default}): ").strip()
    return int(raw) if raw else default


def interactive_color(image: np.ndarray, out_dir: Path) -> None:
    """
    Provide a CLI menu for color operations and save the chosen result.

    Args:
        image: Current working BGR image.
        out_dir: Directory for interactive outputs.
    """
    ensure_output_dir(out_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    menu = """
[COLOR]
 1) Brightness
 2) Contrast
 3) Brightness+Contrast
 4) Log mapping
 5) Exponential mapping
 6) Histogram equalization
 7) Histogram specification
 0) Quay lại
"""
    print(menu)
    choice = input("Chọn: ").strip()
    if choice == "0":
        return

    out_name = input("Tên file (không .png): ").strip() or "color_result"
    if choice == "1":
        b = _input_float("b", 40.0)
        result = brightness_adjust(image, b=b)
    elif choice == "2":
        a = _input_float("a", 1.4)
        result = contrast_adjust(image, a=a)
    elif choice == "3":
        a = _input_float("a", 1.2)
        b = _input_float("b", 25.0)
        result = brightness_contrast_adjust(image, a=a, b=b)
    elif choice == "4":
        c = _input_float("c", 1.0)
        result = log_mapping(gray, c=c)
    elif choice == "5":
        c = _input_float("c", 1.2)
        result = exp_mapping(gray, c=c)
    elif choice == "6":
        result = histogram_equalization(gray)
    elif choice == "7":
        heq = histogram_equalization(gray)
        result = histogram_specification(gray, heq)
    else:
        print("Lựa chọn không hợp lệ.")
        return

    cv2.imwrite(str((out_dir / f"{out_name}.png")), result)
    print(f"[COLOR] Đã lưu {out_dir / (out_name + '.png')}")


def interactive_geometry(image: np.ndarray, out_dir: Path) -> None:
    """
    Provide a CLI menu for geometric operations and save the result.

    Args:
        image: Current working BGR image.
        out_dir: Directory for interactive outputs.
    """
    ensure_output_dir(out_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    menu = """
[GEOMETRY]
 1) Scale
 2) Rotate
 3) Translate
 4) Shear
 0) Quay lại
"""
    print(menu)
    choice = input("Chọn: ").strip()
    if choice == "0":
        return

    out_name = input("Tên file (không .png): ").strip() or "geo_result"
    if choice == "1":
        sx = _input_float("sx", 1.5)
        sy = _input_float("sy", 1.5)
        A, out_size = build_scale_matrix(gray, sx=sx, sy=sy)
    elif choice == "2":
        angle = _input_float("góc (độ)", 30.0)
        A, out_size = build_rotation_matrix(gray, angle_deg=angle)
    elif choice == "3":
        tx = _input_float("tx", 40.0)
        ty = _input_float("ty", 25.0)
        A, out_size = build_translation_matrix(gray, tx=tx, ty=ty)
    elif choice == "4":
        kx = _input_float("kx", 0.4)
        ky = _input_float("ky", 0.0)
        A, out_size = build_shear_matrix(gray, kx=kx, ky=ky)
    else:
        print("Lựa chọn không hợp lệ.")
        return

    interp = input("Nội suy [1] nearest, [2] bilinear (default=2): ").strip() or "2"
    if interp == "1":
        result = warp_affine_nearest(gray, A, out_size)
    else:
        result = warp_affine_bilinear(gray, A, out_size)

    cv2.imwrite(str(out_dir / f"{out_name}.png"), result)
    print(f"[GEOMETRY] Đã lưu {out_dir / (out_name + '.png')}")


def run_mode_interactive(out_dir: Path) -> None:
    """
    Start the interactive loop where the user can pick algorithms repeatedly.

    Args:
        out_dir: Root directory used to store interactive results.
    """
    ensure_output_dir(out_dir)
    image, source_desc = choose_image_interactively()
    print(f"[INTERACTIVE] Đang dùng ảnh: {source_desc}")

    while True:
        menu = """
======== MENU INTERACTIVE ========
 1) Color transformations
 2) Geometric transformations
 9) Đổi ảnh đầu vào
 0) Thoát
"""
        print(menu)
        choice = input("Chọn: ").strip()
        if choice == "0":
            print("Thoát interactive mode.")
            break
        if choice == "9":
            image, source_desc = choose_image_interactively()
            print(f"[INTERACTIVE] Đổi ảnh thành: {source_desc}")
            continue
        if choice == "1":
            interactive_color(image, out_dir / "interactive_color")
        elif choice == "2":
            interactive_geometry(image, out_dir / "interactive_geometry")
        else:
            print("Lựa chọn không hợp lệ.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Program entry point handling the selected run mode."""
    args = parse_args()
    ensure_output_dir(args.out_dir)

    if args.mode == "test":
        run_mode_test(args.out_dir)
    elif args.mode == "batch":
        run_mode_batch(args.image, args.out_dir)
    elif args.mode == "interactive":
        run_mode_interactive(args.out_dir)
    else:
        raise ValueError(f"Mode không hợp lệ: {args.mode}")


if __name__ == "__main__":
    main()
