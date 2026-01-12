from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from smoothing import SpatialSmoothing
from utils import FileIOHandle, add_noise, ensure_dir, save_table_grid

LENA_URL = "http://www.ess.ic.kanagawa-it.ac.jp/std_img/colorimage/Lenna.jpg"
DEFAULT_LENA_PATH = Path("assets") / "Lenna.jpg"
FALLBACK_LENNA_PATH = Path("assets") / "Lenna.jpg"


class SmoothingApp:
    """Application wrapper to run smoothing."""

    KERNEL_SIZES = [3, 5, 7]
    GAUSSIAN_SIGMA = {"sp": 1.0, "gaussian": 12.0}
    COND_DIFF_THRESHOLD = 20
    COND_RANGE_PARAMS = {
        "sp": {"ksize": 3, "low": 5, "high": 250},
        "gaussian": {"ksize": 3, "low": 60, "high": 200},
    }
    MMSE_NOISE_VAR = 20.0
    NOISE_CONFIG = {
        "sp": {"mode": "s&p", "amount": 0.05},
        "gaussian": {"mode": "gaussian", "amount": 12.0},
    }

    def __init__(self) -> None:
        """Initialize app and parse arguments."""
        self.args = self._parse_args()

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        """Parse CLI arguments for smoothing demo.

        Returns:
            argparse.Namespace: Parsed CLI arguments.
        """
        parser = argparse.ArgumentParser(description="Spatial smoothing filters demo")
        parser.add_argument(
            "--image",
            type=Path,
            help="Input image path (omit to use assets/Lena.jpg)",
        )
        parser.add_argument(
            "--out_dir",
            type=Path,
            default=Path("outputs"),
            help="Output directory for results",
        )
        return parser.parse_args()

    @staticmethod
    def _fetch_default_lena() -> Path:
        """Ensure default Lena image exists locally; download if missing.

        Returns:
            Path: Path to the Lena image on disk.
        """
        ensure_dir(DEFAULT_LENA_PATH.parent)
        if DEFAULT_LENA_PATH.exists():
            return DEFAULT_LENA_PATH
        if FALLBACK_LENNA_PATH.exists():
            return FALLBACK_LENNA_PATH
        print(f"[INFO] Downloading Lena image to {DEFAULT_LENA_PATH}")
        urllib.request.urlretrieve(LENA_URL, DEFAULT_LENA_PATH)
        return DEFAULT_LENA_PATH

    @staticmethod
    def _load_image(path: Path) -> cv2.Mat:
        """Read image as BGR; raise if not found.

        Args:
            path: Path to image.

        Returns:
            cv2.Mat: Loaded BGR image.

        Raises:
            FileNotFoundError: When the image cannot be read.
        """
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {path}")
        return img

    def _choose_image(self) -> Path:
        """Resolve image path: CLI provided or default Lena.

        Returns:
            Path: Resolved image path.
        """
        return (
            self.args.image
            if self.args.image is not None
            else self._fetch_default_lena()
        )

    def _run_pipeline(self, image_path: Path, out_dir: Path) -> None:
        """Add noise, run all smoothing filters, and write outputs + comparisons.

        Args:
            image_path: Path to input image.
            out_dir: Output directory for all results.
        """
        print(f"\n{'='*60}")
        print(f"[INFO] Starting processing: {image_path}")
        print(f"{'='*60}\n")

        file_io = FileIOHandle(out_dir)
        dirs = file_io.paths
        print("[STEP 1/6] Reading and preparing image...")
        image = self._load_image(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(dirs["pre"] / "original_color.png"), image)
        cv2.imwrite(str(dirs["pre"] / "original_gray.png"), gray)
        print("  ✓ Saved original_color.png and original_gray.png")

        # Create both noise types
        print("\n[STEP 2/6] Generating noise...")
        noisy_sp = add_noise(gray, **self.NOISE_CONFIG["sp"])
        noisy_gauss = add_noise(gray, **self.NOISE_CONFIG["gaussian"])
        cv2.imwrite(str(dirs["pre"] / "noisy_salt_pepper.png"), noisy_sp)
        cv2.imwrite(str(dirs["pre"] / "noisy_gaussian.png"), noisy_gauss)
        print("  ✓ Generated and saved noisy_salt_pepper.png")
        print("  ✓ Generated and saved noisy_gaussian.png")

        # Process filters on both noise types; save singles first, then tables
        print("\n[STEP 3/6] Running filters and saving individual outputs...")

        # Mean filter: 3x3, 5x5, 7x7
        print("  [1/8] Mean filter (3x3, 5x5, 7x7)...")
        mean_rows = []
        for ksize in self.KERNEL_SIZES:
            mean_sp = SpatialSmoothing.mean(noisy_sp, ksize)
            mean_gauss = SpatialSmoothing.mean(noisy_gauss, ksize)
            file_io.save_single_image(mean_sp, "sp", "mean", f"{ksize}x{ksize}")
            file_io.save_single_image(
                mean_gauss, "gaussian", "mean", f"{ksize}x{ksize}"
            )
            mean_rows.append((f"{ksize}x{ksize}", mean_sp, mean_gauss))
        save_table_grid(mean_rows, dirs["result"] / "table_mean_filter.png")
        print("    ✓ Completed Mean filter")

        # Gaussian filter: 3x3, 5x5, 7x7
        print("  [2/8] Gaussian filter (3x3, 5x5, 7x7)...")
        gauss_rows = []
        for ksize in self.KERNEL_SIZES:
            gauss_sp = SpatialSmoothing.gaussian(
                noisy_sp, ksize, self.GAUSSIAN_SIGMA["sp"]
            )
            gauss_gauss = SpatialSmoothing.gaussian(
                noisy_gauss, ksize, self.GAUSSIAN_SIGMA["gaussian"]
            )
            file_io.save_single_image(gauss_sp, "sp", "gaussian", f"{ksize}x{ksize}")
            file_io.save_single_image(
                gauss_gauss, "gaussian", "gaussian", f"{ksize}x{ksize}"
            )
            gauss_rows.append((f"{ksize}x{ksize}", gauss_sp, gauss_gauss))
        save_table_grid(gauss_rows, dirs["result"] / "table_gaussian_filter.png")
        print("    ✓ Completed Gaussian filter")

        # Median filter: 3x3, 5x5, 7x7
        print("  [3/8] Median filter (3x3, 5x5, 7x7)...")
        median_rows = []
        for ksize in self.KERNEL_SIZES:
            median_sp = SpatialSmoothing.median(noisy_sp, ksize)
            median_gauss = SpatialSmoothing.median(noisy_gauss, ksize)
            file_io.save_single_image(median_sp, "sp", "median", f"{ksize}x{ksize}")
            file_io.save_single_image(
                median_gauss, "gaussian", "median", f"{ksize}x{ksize}"
            )
            median_rows.append((f"{ksize}x{ksize}", median_sp, median_gauss))
        save_table_grid(median_rows, dirs["result"] / "table_median_filter.png")
        print("    ✓ Completed Median filter")

        # Conditional (difference-based): 3x3, 5x5, 7x7
        print("  [4/8] Conditional (difference-based, 3x3, 5x5, 7x7)...")
        cond_diff_rows = []
        for ksize in self.KERNEL_SIZES:
            cond_diff_sp = SpatialSmoothing.conditional(
                noisy_sp, ksize, threshold=self.COND_DIFF_THRESHOLD
            )
            cond_diff_gauss = SpatialSmoothing.conditional(
                noisy_gauss, ksize, threshold=self.COND_DIFF_THRESHOLD
            )
            label = f"{ksize}x{ksize}"
            file_io.save_single_image(cond_diff_sp, "sp", "conditional_diff", label)
            file_io.save_single_image(
                cond_diff_gauss, "gaussian", "conditional_diff", label
            )
            cond_diff_rows.append((label, cond_diff_sp, cond_diff_gauss))
        save_table_grid(
            cond_diff_rows,
            dirs["result"] / "table_conditional_diff.png",
        )
        print("    ✓ Completed Conditional (difference-based)")

        # Conditional averaging (range-based): 3x3, 5x5, 7x7
        print("  [5/8] Conditional averaging (range-based, 3x3, 5x5, 7x7)...")
        cond_range_rows = []
        for ksize in self.KERNEL_SIZES:
            cond_range_sp = SpatialSmoothing.conditional_range_impulse(
                noisy_sp,
                ksize,
                self.COND_RANGE_PARAMS["sp"]["low"],
                self.COND_RANGE_PARAMS["sp"]["high"],
            )
            cond_range_gauss = SpatialSmoothing.conditional_range_smooth(
                noisy_gauss,
                ksize,
                self.COND_RANGE_PARAMS["gaussian"]["low"],
                self.COND_RANGE_PARAMS["gaussian"]["high"],
            )
            label = f"{ksize}x{ksize}"
            file_io.save_single_image(cond_range_sp, "sp", "conditional_range", label)
            file_io.save_single_image(
                cond_range_gauss, "gaussian", "conditional_range", label
            )
            cond_range_rows.append((label, cond_range_sp, cond_range_gauss))
        save_table_grid(
            cond_range_rows,
            dirs["result"] / "table_conditional_range.png",
        )
        print("    ✓ Completed Conditional averaging")

        # Gradient-weighted: 3x3, 5x5, 7x7
        print("  [6/8] Gradient-weighted averaging (3x3, 5x5, 7x7)...")
        grad_rows = []
        for ksize in self.KERNEL_SIZES:
            grad_sp = SpatialSmoothing.gradient_weighted(noisy_sp, ksize)
            grad_gauss = SpatialSmoothing.gradient_weighted(noisy_gauss, ksize)
            label = f"{ksize}x{ksize}"
            file_io.save_single_image(grad_sp, "sp", "gradient_weighted", label)
            file_io.save_single_image(
                grad_gauss, "gaussian", "gradient_weighted", label
            )
            grad_rows.append((label, grad_sp, grad_gauss))
        save_table_grid(
            grad_rows,
            dirs["result"] / "table_gradient_weighted.png",
        )
        print("    ✓ Completed Gradient-weighted averaging")

        # Rotating mask: 3x3
        print("  [7/8] Rotating mask (3x3)...")
        rot_rows = []
        for ksize in self.KERNEL_SIZES[0:1]:
            rot_sp = SpatialSmoothing.rotating_mask(noisy_sp, ksize)
            rot_gauss = SpatialSmoothing.rotating_mask(noisy_gauss, ksize)
            label = f"{ksize}x{ksize}"
            file_io.save_single_image(rot_sp, "sp", "rotating_mask", label)
            file_io.save_single_image(rot_gauss, "gaussian", "rotating_mask", label)
            rot_rows.append((label, rot_sp, rot_gauss))
        save_table_grid(
            rot_rows,
            dirs["result"] / "table_rotating_mask.png",
        )
        print("    ✓ Completed Rotating mask")

        # MMSE: 3x3, 5x5, 7x7
        print("  [8/8] MMSE filter (3x3, 5x5, 7x7)...")
        mmse_rows = []
        for ksize in self.KERNEL_SIZES:
            mmse_sp = SpatialSmoothing.mmse(
                noisy_sp, ksize, noise_variance=self.MMSE_NOISE_VAR
            )
            mmse_gauss = SpatialSmoothing.mmse(
                noisy_gauss, ksize, noise_variance=self.MMSE_NOISE_VAR
            )
            label = f"{ksize}x{ksize}"
            file_io.save_single_image(mmse_sp, "sp", "mmse", label)
            file_io.save_single_image(mmse_gauss, "gaussian", "mmse", label)
            mmse_rows.append((label, mmse_sp, mmse_gauss))
        save_table_grid(
            mmse_rows,
            dirs["result"] / "table_mmse_filter.png",
        )
        print("    ✓ Completed MMSE filter")

        print("\n[STEP 4/6] Generated and saved all individual results.")

        # Generate OpenCV baselines for comparison (all 3x3)
        print("\n[STEP 5/6] Generating OpenCV baselines for comparison...")
        cv_blur3_sp = cv2.blur(noisy_sp, (3, 3))
        cv_blur3_gauss = cv2.blur(noisy_gauss, (3, 3))
        cv_gauss3_sp = cv2.GaussianBlur(noisy_sp, (3, 3), 1.0)
        cv_gauss3_gauss = cv2.GaussianBlur(noisy_gauss, (3, 3), 1.0)
        cv_median3_sp = cv2.medianBlur(noisy_sp, 3)
        cv_median3_gauss = cv2.medianBlur(noisy_gauss, 3)
        cv_bilateral_sp = cv2.bilateralFilter(noisy_sp, 3, 20, 10)
        cv_bilateral_gauss = cv2.bilateralFilter(noisy_gauss, 3, 20, 10)

        # Save baseline singles for completeness
        file_io.save_single_image(cv_blur3_sp, "sp", "cv_blur", "3x3")
        file_io.save_single_image(cv_blur3_gauss, "gaussian", "cv_blur", "3x3")
        file_io.save_single_image(cv_gauss3_sp, "sp", "cv_gaussian", "3x3")
        file_io.save_single_image(cv_gauss3_gauss, "gaussian", "cv_gaussian", "3x3")
        file_io.save_single_image(cv_median3_sp, "sp", "cv_median", "3x3")
        file_io.save_single_image(cv_median3_gauss, "gaussian", "cv_median", "3x3")
        file_io.save_single_image(cv_bilateral_sp, "sp", "cv_bilateral", "3x3")
        file_io.save_single_image(cv_bilateral_gauss, "gaussian", "cv_bilateral", "3x3")
        print("  ✓ Generated and saved OpenCV baselines")

        # Build comparison tables after all outputs exist
        print(
            "\n[STEP 6/6] Creating comparison tables with all filters and OpenCV baselines..."
        )

        # Mean vs blur
        compare_mean_vs_blur = [
            ("3x3", mean_rows[0][1], mean_rows[0][2]),  # custom mean
            ("cv2.blur", cv_blur3_sp, cv_blur3_gauss),  # OpenCV blur
        ]
        save_table_grid(
            compare_mean_vs_blur,
            dirs["compare"] / "compare_mean_vs_blur.png",
            header_labels=("Method", "Salt & Pepper", "Gaussian"),
        )

        # Gaussian vs GaussianBlur
        compare_gauss_vs_gauss = [
            ("3x3", gauss_rows[0][1], gauss_rows[0][2]),  # custom gaussian 3x3
            (
                "cv2.GaussianBlur",
                cv_gauss3_sp,
                cv_gauss3_gauss,
            ),  # OpenCV GaussianBlur 3x3
        ]
        save_table_grid(
            compare_gauss_vs_gauss,
            dirs["compare"] / "compare_gaussian_vs_gaussian.png",
            header_labels=("Method", "Salt & Pepper", "Gaussian"),
        )

        # Median vs medianBlur
        compare_median_vs_median = [
            ("3x3", median_rows[0][1], median_rows[0][2]),  # custom median 3x3
            (
                "cv2.medianBlur",
                cv_median3_sp,
                cv_median3_gauss,
            ),  # OpenCV medianBlur 3x3
        ]
        save_table_grid(
            compare_median_vs_median,
            dirs["compare"] / "compare_median_vs_median.png",
            header_labels=("Method", "Salt & Pepper", "Gaussian"),
        )

        # Group 2: All filters comparison - Salt & Pepper noise (use 3x3 outputs for compare)
        mean_sp = mean_rows[0][1]
        gauss_sp = gauss_rows[0][1]
        median_sp = median_rows[0][1]
        cond_diff_sp = cond_diff_rows[0][1]
        cond_range_sp = cond_range_rows[0][1]
        grad_sp = grad_rows[0][1]
        rot_sp = rot_rows[0][1]
        compare_all_sp = [
            ("Mean 3x3", mean_sp),
            ("Gaussian 3x3", gauss_sp),
            ("Median 3x3", median_sp),
            ("Conditional (diff)", cond_diff_sp),
            ("Conditional (range)", cond_range_sp),
            ("Gradient-weighted", grad_sp),
            ("Rotating mask", rot_sp),
            ("cv2.GaussianBlur", cv_gauss3_sp),
            ("cv2.bilateralFilter", cv_bilateral_sp),
            ("cv2.medianBlur", cv_median3_sp),
        ]
        save_table_grid(
            compare_all_sp,
            dirs["compare"] / "compare_all_filters_salt_pepper.png",
            header_labels=("Method", "Salt & Pepper", "Salt & Pepper"),
            cols=5,
        )

        # Group 2: All filters comparison - Gaussian noise (use 3x3 outputs for compare)
        mean_gauss = mean_rows[0][2]
        gauss_gauss = gauss_rows[0][2]
        median_gauss = median_rows[0][2]
        cond_diff_gauss = cond_diff_rows[0][2]
        cond_range_gauss = cond_range_rows[0][2]
        grad_gauss = grad_rows[0][2]
        rot_gauss = rot_rows[0][2]
        compare_all_gauss = [
            ("Mean 3x3", mean_gauss),
            ("Gaussian 3x3", gauss_gauss),
            ("Median 3x3", median_gauss),
            ("Conditional (diff)", cond_diff_gauss),
            ("Conditional (range)", cond_range_gauss),
            ("Gradient-weighted", grad_gauss),
            ("Rotating mask", rot_gauss),
            ("cv2.GaussianBlur", cv_gauss3_gauss),
            ("cv2.bilateralFilter", cv_bilateral_gauss),
            ("cv2.medianBlur", cv_median3_gauss),
        ]
        save_table_grid(
            compare_all_gauss,
            dirs["compare"] / "compare_all_filters_gaussian.png",
            header_labels=("Method", "Gaussian", "Gaussian"),
            cols=5,
        )

        # Conditional variants comparison (4 outputs: 2 methods x 2 noises)
        compare_conditional_variants = [
            ("Conditional (diff)", cond_diff_sp, cond_diff_gauss),
            ("Conditional (range)", cond_range_sp, cond_range_gauss),
        ]
        save_table_grid(
            compare_conditional_variants,
            dirs["compare"] / "compare_conditional_variants.png",
            header_labels=("Method", "Salt & Pepper", "Gaussian"),
        )

        # Group 3: MMSE vs GaussianBlur
        compare_mmse = [
            ("MMSE 5x5", mmse_sp, mmse_gauss),
            (
                "cv2.GaussianBlur",
                cv_gauss3_sp,
                cv_gauss3_gauss,
            ),  # OpenCV GaussianBlur 3x3
        ]
        save_table_grid(
            compare_mmse,
            dirs["compare"] / "compare_mmse_vs_gaussian.png",
            header_labels=("Method", "Salt & Pepper", "Gaussian"),
        )

        print("  ✓ Created all comparison tables with all filters and OpenCV baselines")

        print("\n[COMPLETED] All results have been saved!")
        print(f"\n{'='*60}")
        print(f"✓ Processed: {image_path}")
        print(f"✓ Results saved in: {out_dir.resolve()}")
        print(f"  - Result tables: {dirs['result']}")
        print(f"  - Comparison tables: {dirs['compare']}")
        print(f"{'='*60}\n")

    def run(self) -> None:
        """Execute the app."""
        image_path = self._choose_image()
        self._run_pipeline(image_path, self.args.out_dir)


def main() -> None:
    """Entry point: instantiate and run the smoothing app."""
    app = SmoothingApp()
    app.run()


if __name__ == "__main__":
    main()
