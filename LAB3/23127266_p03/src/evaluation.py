"""Evaluation script for all smoothing filters with quantitative metrics."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from smoothing import SpatialSmoothing
from utils import add_noise, ensure_dir

# OpenCV configurations matching main.py
CV_CONFIGS = {
    "cv2.blur": {"ksize": (3, 3)},
    "cv2.GaussianBlur": {"ksize": (3, 3), "sigmaX": 1.0},
    "cv2.medianBlur": {"ksize": 3},
    "cv2.bilateralFilter": {"d": 5, "sigmaColor": 20, "sigmaSpace": 10},
}

# Map filter names in code to directory names in result/
FILTER_NAME_MAP = {
    "mean": "mean",
    "gaussian": "gaussian",
    "median": "median",
    "conditional_range": "conditional_range",
    "conditional_diff": "conditional_diff",
    "gradient_weighted": "gradient_weighted",
    "rotating_mask": "rotating_mask",
    "mmse": "mmse",
    "cv2.blur": "cv_blur",
    "cv2.GaussianBlur": "cv_gaussian",
    "cv2.medianBlur": "cv_median",
    "cv2.bilateralFilter": "cv_bilateral",
}


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images.

    Args:
        image1: First image.
        image2: Second image.

    Returns:
        float: MSE value.
    """
    return float(np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2))


def psnr(image1: np.ndarray, image2: np.ndarray, max_pixel: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        image1: First image.
        image2: Second image.
        max_pixel: Maximum pixel value (default 255).

    Returns:
        float: PSNR value in dB.
    """
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return float("inf")
    return float(20 * np.log10(max_pixel / np.sqrt(mse_val)))


def detect_edges(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges in image using Canny or Sobel.

    Args:
        image: Input grayscale image.
        method: 'canny' or 'sobel'.

    Returns:
        np.ndarray: Binary edge map.
    """
    if method == "canny":
        return cv2.Canny(image, 50, 150)
    elif method == "sobel":
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        return (sobel > 50).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_edge_preservation(
    original: np.ndarray, filtered: np.ndarray, method: str = "canny"
) -> dict[str, float]:
    """Evaluate edge preservation capability.

    Args:
        original: Original image.
        filtered: Filtered image.
        method: Edge detection method ('canny' or 'sobel').

    Returns:
        dict[str, float]: Edge preservation metrics.
    """
    edges_gt = detect_edges(original, method)
    edges_filtered = detect_edges(filtered, method)

    # Count edge pixels
    edge_count_gt = np.sum(edges_gt > 0)
    edge_count_filtered = np.sum(edges_filtered > 0)

    # Edge preservation ratio
    if edge_count_gt > 0:
        edge_ratio = edge_count_filtered / edge_count_gt
    else:
        edge_ratio = 0.0

    # Edge overlap (intersection over union)
    intersection = np.sum((edges_gt > 0) & (edges_filtered > 0))
    union = np.sum((edges_gt > 0) | (edges_filtered > 0))
    iou = intersection / union if union > 0 else 0.0

    return {
        "edge_count_original": float(edge_count_gt),
        "edge_count_filtered": float(edge_count_filtered),
        "edge_ratio": float(edge_ratio),
        "edge_iou": float(iou),
    }


def load_filtered_image(
    result_dir: Path, filter_name: str, noise_type: str, ksize: int = 3
) -> np.ndarray | None:
    """Load filtered image from result directory.

    Args:
        result_dir: Root result directory (outputs/result).
        filter_name: Filter name in code.
        noise_type: 'sp' or 'gaussian'.
        ksize: Kernel size (default 3).

    Returns:
        np.ndarray: Loaded image, or None if not found.
    """
    dir_name = FILTER_NAME_MAP.get(filter_name)
    if dir_name is None:
        return None

    label = f"{ksize}x{ksize}"
    image_path = (
        result_dir / noise_type / dir_name / f"{dir_name}_{noise_type}_{label}.png"
    )

    if not image_path.exists():
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return image


def format_filter_info(name: str, filter_type: str, **kwargs) -> str:
    """Format filter information string.

    Args:
        name: Filter name.
        filter_type: 'custom' or 'opencv'.
        **kwargs: Filter parameters.

    Returns:
        str: Formatted filter information.
    """
    info_parts = [f"Filter: {name}", f"Type: {filter_type}", f"Kernel: 3x3"]

    if filter_type == "custom":
        if name == "gaussian":
            info_parts.append(f"Sigma: {kwargs.get('sigma', 1.0)}")
        elif name.startswith("conditional_range"):
            info_parts.append(
                f"Range: [{kwargs.get('low', 80)}, {kwargs.get('high', 200)}]"
            )
        elif name == "conditional_diff":
            info_parts.append(f"Threshold: {kwargs.get('threshold', 20)}")
        elif name == "mmse":
            info_parts.append(f"Noise variance: {kwargs.get('noise_variance', 20.0)}")
    elif filter_type == "opencv":
        config = CV_CONFIGS.get(name, {})
        if name == "cv2.GaussianBlur":
            info_parts.append(f"Sigma: {config.get('sigmaX', 1.0)}")
        elif name == "cv2.bilateralFilter":
            info_parts.append(
                f"d={config.get('d', 5)}, sigmaColor={config.get('sigmaColor', 20)}, sigmaSpace={config.get('sigmaSpace', 10)}"
            )

    return " | ".join(info_parts)


def main() -> None:
    """Run evaluation for all filters."""
    print("\n" + "=" * 80)
    print("EVALUATION: All Smoothing Filters (Kernel 3x3)")
    print("=" * 80 + "\n")

    # Setup directories
    eval_dir = Path("outputs") / "eval"
    ensure_dir(eval_dir)
    result_dir = Path("outputs") / "result"

    if not result_dir.exists():
        print(f"[ERROR] Result directory not found: {result_dir}")
        print("Please run main.py first to generate filtered images.")
        return

    # Load original image (needed for metrics calculation)
    lena_path = Path("assets") / "Lena.jpg"
    if not lena_path.exists():
        lena_path = Path("assets") / "Lenna.jpg"

    # Try loading from preprocessingImage if assets not found
    if not lena_path.exists():
        preprocess_dir = Path("outputs") / "preprocessingImage"
        original_path = preprocess_dir / "original_gray.png"
        if original_path.exists():
            print(f"[INFO] Loading original from: {original_path}")
            gray = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[ERROR] Cannot read original image from preprocessingImage")
                return
        else:
            print(
                f"[ERROR] Cannot find original image. Please ensure assets/Lena.jpg or preprocessingImage/original_gray.png exists"
            )
            return
    else:
        print(f"[INFO] Loading image: {lena_path}")
        image = cv2.imread(str(lena_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[ERROR] Cannot read image")
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Image size: {gray.shape}")
    print(f"[INFO] Loading filtered images from: {result_dir}")

    # Save original for reference
    cv2.imwrite(str(eval_dir / "original.png"), gray)

    # Define all filters to test
    custom_filters = [
        ("mean", {"ksize": 3}),
        ("gaussian", {"ksize": 3, "sigma": 1.0}),
        ("median", {"ksize": 3}),
        ("conditional_range_impulse", {"ksize": 3, "low": 5, "high": 250}),
        ("conditional_range_smooth", {"ksize": 3, "low": 60, "high": 200}),
        ("conditional_diff", {"ksize": 3, "threshold": 20}),
        ("gradient_weighted", {"ksize": 3}),
        ("rotating_mask", {"ksize": 3}),
        ("mmse", {"ksize": 5, "noise_variance": 20.0}),  # MMSE uses 5x5
    ]

    opencv_filters = list(CV_CONFIGS.keys())

    # Results storage (use dict to prevent duplicates)
    results_dict = {}
    results = []

    print("\n" + "=" * 80)
    print("RUNNING CUSTOM FILTERS")
    print("=" * 80)

    # Run custom filters
    for filter_name, params in custom_filters:
        # Skip if already processed
        if filter_name in results_dict:
            print(f"\n[{filter_name.upper()}] - SKIPPED (duplicate)")
            continue

        print(f"\n[{filter_name.upper()}]")
        info = format_filter_info(filter_name, "custom", **params)
        print(f"  {info}")

        try:
            # Measure computation time
            start_time = time.perf_counter()

            # Load filtered images from result directory
            ksize = params.get("ksize", 3)
            filtered_sp = load_filtered_image(result_dir, filter_name, "sp", ksize)
            filtered_gauss = load_filtered_image(
                result_dir, filter_name, "gaussian", ksize
            )

            if filtered_sp is None or filtered_gauss is None:
                print(f"  ✗ Cannot load filtered images for {filter_name}")
                continue

            # Calculate metrics
            mse_sp = mse(gray, filtered_sp)
            psnr_sp = psnr(gray, filtered_sp)
            mse_gauss = mse(gray, filtered_gauss)
            psnr_gauss = psnr(gray, filtered_gauss)

            # Edge preservation
            edge_sp = evaluate_edge_preservation(gray, filtered_sp)
            edge_gauss = evaluate_edge_preservation(gray, filtered_gauss)

            computation_time = time.perf_counter() - start_time

            # Save images to eval directory
            cv2.imwrite(str(eval_dir / f"{filter_name}_salt_pepper.png"), filtered_sp)
            cv2.imwrite(str(eval_dir / f"{filter_name}_gaussian.png"), filtered_gauss)

            result_entry = {
                "name": filter_name,
                "type": "custom",
                "info": info,
                "mse_sp": mse_sp,
                "psnr_sp": psnr_sp,
                "mse_gauss": mse_gauss,
                "psnr_gauss": psnr_gauss,
                "edge_sp": edge_sp,
                "edge_gauss": edge_gauss,
                "execution_time": computation_time,
            }
            results_dict[filter_name] = result_entry
            results.append(result_entry)

            print(f"  ✓ MSE (S&P): {mse_sp:.2f} | PSNR (S&P): {psnr_sp:.2f} dB")
            print(
                f"  ✓ MSE (Gauss): {mse_gauss:.2f} | PSNR (Gauss): {psnr_gauss:.2f} dB"
            )
            print(
                f"  ✓ Edge IOU (S&P): {edge_sp['edge_iou']:.3f} | Edge IOU (Gauss): {edge_gauss['edge_iou']:.3f}"
            )
            print(f"  ✓ Computation time: {computation_time*1000:.2f} ms")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("RUNNING OPENCV FILTERS")
    print("=" * 80)

    # Run OpenCV filters
    for filter_name in opencv_filters:
        # Skip if already processed
        if filter_name in results_dict:
            print(f"\n[{filter_name.upper()}] - SKIPPED (duplicate)")
            continue

        print(f"\n[{filter_name.upper()}]")
        info = format_filter_info(filter_name, "opencv")
        print(f"  {info}")

        try:
            # Measure computation time
            start_time = time.perf_counter()

            # Load filtered images from result directory
            filtered_sp = load_filtered_image(result_dir, filter_name, "sp", 3)
            filtered_gauss = load_filtered_image(result_dir, filter_name, "gaussian", 3)

            if filtered_sp is None or filtered_gauss is None:
                print(f"  ✗ Cannot load filtered images for {filter_name}")
                continue

            # Calculate metrics
            mse_sp = mse(gray, filtered_sp)
            psnr_sp = psnr(gray, filtered_sp)
            mse_gauss = mse(gray, filtered_gauss)
            psnr_gauss = psnr(gray, filtered_gauss)

            # Edge preservation
            edge_sp = evaluate_edge_preservation(gray, filtered_sp)
            edge_gauss = evaluate_edge_preservation(gray, filtered_gauss)

            computation_time = time.perf_counter() - start_time

            # Save images to eval directory
            safe_name = filter_name.replace(".", "_")
            cv2.imwrite(str(eval_dir / f"{safe_name}_salt_pepper.png"), filtered_sp)
            cv2.imwrite(str(eval_dir / f"{safe_name}_gaussian.png"), filtered_gauss)

            result_entry = {
                "name": filter_name,
                "type": "opencv",
                "info": info,
                "mse_sp": mse_sp,
                "psnr_sp": psnr_sp,
                "mse_gauss": mse_gauss,
                "psnr_gauss": psnr_gauss,
                "edge_sp": edge_sp,
                "edge_gauss": edge_gauss,
                "execution_time": computation_time,
            }
            results_dict[filter_name] = result_entry
            results.append(result_entry)

            print(f"  ✓ MSE (S&P): {mse_sp:.2f} | PSNR (S&P): {psnr_sp:.2f} dB")
            print(
                f"  ✓ MSE (Gauss): {mse_gauss:.2f} | PSNR (Gauss): {psnr_gauss:.2f} dB"
            )
            print(
                f"  ✓ Edge IOU (S&P): {edge_sp['edge_iou']:.3f} | Edge IOU (Gauss): {edge_gauss['edge_iou']:.3f}"
            )
            print(f"  ✓ Computation time: {computation_time*1000:.2f} ms")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Print summary table (metrics)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - METRICS")
    print("=" * 80)
    print(
        f"{'Filter':<25} {'MSE (S&P)':<12} {'PSNR (S&P)':<12} {'MSE (Gauss)':<14} {'PSNR (Gauss)':<14} {'Edge IOU (S&P)':<16} {'Edge IOU (Gauss)':<16}"
    )
    print("-" * 112)

    # Sort by PSNR (Gaussian) descending
    results_sorted = sorted(results, key=lambda x: x["psnr_gauss"], reverse=True)

    for r in results_sorted:
        print(
            f"{r['name']:<25} {r['mse_sp']:<12.2f} {r['psnr_sp']:<12.2f} {r['mse_gauss']:<14.2f} {r['psnr_gauss']:<14.2f} {r['edge_sp']['edge_iou']:<16.3f} {r['edge_gauss']['edge_iou']:<16.3f}"
        )

    # Print computation time table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - COMPUTATION TIME")
    print("=" * 80)
    print(f"{'Filter':<25} {'Time (ms)':<15} {'Time (s)':<12} {'Rank':<8}")
    print("-" * 62)

    # Sort by computation time ascending (fastest first)
    results_time_sorted = sorted(results, key=lambda x: x["execution_time"])

    for idx, r in enumerate(results_time_sorted, 1):
        time_ms = r["execution_time"] * 1000
        time_s = r["execution_time"]
        print(f"{r['name']:<25} {time_ms:<15.2f} {time_s:<12.4f} {idx:<8}")

    # Calculate statistics
    total_time = sum(r["execution_time"] for r in results)
    avg_time = total_time / len(results) if results else 0
    fastest = min(results, key=lambda x: x["execution_time"])
    slowest = max(results, key=lambda x: x["execution_time"])

    print("\n" + "-" * 70)
    print(f"Total computation time: {total_time:.4f} s ({total_time*1000:.2f} ms)")
    print(f"Average computation time: {avg_time:.4f} s ({avg_time*1000:.2f} ms)")
    print(f"Fastest: {fastest['name']} ({fastest['execution_time']*1000:.2f} ms)")
    print(f"Slowest: {slowest['name']} ({slowest['execution_time']*1000:.2f} ms)")

    print("\n" + "=" * 80)
    print(f"[COMPLETED] All results saved to: {eval_dir.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
