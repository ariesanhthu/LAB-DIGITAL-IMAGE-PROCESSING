import os
import time
import tracemalloc
from typing import Callable, Any, Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

from color_transform import (
    brightness_adjust,
    contrast_adjust,
    brightness_contrast_adjust,
    range_linear_mapping,
    histogram_equalization,
)
from geometric_transforms import (
    build_scale_matrix,
    build_rotation_matrix,
    build_shear_matrix,
    warp_affine_nearest,
    warp_affine_bilinear,
)


def _benchmark(
    func: Callable[..., Any],
    *args,
    n_runs: int = 10,
) -> Tuple[float, float]:
    """
    Benchmark 1 hàm: trả về (avg_time_ms, peak_mem_kb).
    Dùng tracemalloc nên kết quả memory là tương đối (Python-level), không phải RSS.
    """
    # warmup nhỏ tránh cold-start
    func(*args)

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        func(*args)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_time_ms = (t1 - t0) * 1000.0 / n_runs
    peak_kb = peak / 1024.0
    return avg_time_ms, peak_kb


def _print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Op':20s} | {'Impl':15s} | {'Time (ms)':>10s} | {'Peak mem (KB)':>13s}")
    print("-" * 80)


def _print_row(op: str, impl: str, t_ms: float, mem_kb: float):
    print(f"{op:20s} | {impl:15s} | {t_ms:10.3f} | {mem_kb:13.1f}")


def eval_color_ops(img_color: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # các tham số test cố định
    b = 40.0
    a = 1.5
    f1, f2, g1, g2 = 50, 180, 0, 255

    _print_header("COLOR TRANSFORM (KHÔNG GỒM LOG/EXP/SPECIFICATION)")
    rows: List[Dict[str, Any]] = []

    # ---- Brightness ----
    t_ms, mem_kb = _benchmark(brightness_adjust, gray, b)
    _print_row("brightness", "manual", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "brightness",
            "impl": "manual",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    def cv_brightness(x):
        return cv2.convertScaleAbs(x, alpha=1.0, beta=b)

    t_ms, mem_kb = _benchmark(cv_brightness, gray)
    _print_row("brightness", "opencv", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "brightness",
            "impl": "opencv",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Contrast ----
    t_ms, mem_kb = _benchmark(contrast_adjust, gray, a)
    _print_row("contrast", "manual", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "contrast",
            "impl": "manual",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    def cv_contrast(x):
        return cv2.convertScaleAbs(x, alpha=a, beta=0.0)

    t_ms, mem_kb = _benchmark(cv_contrast, gray)
    _print_row("contrast", "opencv", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "contrast",
            "impl": "opencv",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Brightness+Contrast ----
    t_ms, mem_kb = _benchmark(brightness_contrast_adjust, gray, a, b)
    _print_row("bright+contrast", "manual", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "bright+contrast",
            "impl": "manual",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    def cv_bc(x):
        return cv2.convertScaleAbs(x, alpha=a, beta=b)

    t_ms, mem_kb = _benchmark(cv_bc, gray)
    _print_row("bright+contrast", "opencv", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "bright+contrast",
            "impl": "opencv",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Range linear mapping [f1,f2] -> [g1,g2] ----
    t_ms, mem_kb = _benchmark(range_linear_mapping, gray, f1, f2, g1, g2)
    _print_row("range_map", "manual", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "range_map",
            "impl": "manual",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # OpenCV version: dùng LUT với cùng công thức nhưng lookup do cv2 thực hiện
    lut = np.arange(256, dtype=np.uint8)
    mask = (lut >= f1) & (lut <= f2)
    if f2 != f1:
        lut = lut.astype(np.float32)
        lut[mask] = g1 + (lut[mask] - f1) * (g2 - g1) / float(f2 - f1)
        lut = np.clip(lut, 0, 255).astype(np.uint8)

    def cv_range_map(x):
        return cv2.LUT(x, lut)

    t_ms, mem_kb = _benchmark(cv_range_map, gray)
    _print_row("range_map", "opencv+LUT", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "range_map",
            "impl": "opencv+LUT",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Histogram equalization ----
    t_ms, mem_kb = _benchmark(histogram_equalization, gray)
    _print_row("hist_equalize", "manual", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "hist_equalize",
            "impl": "manual",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    def cv_hist_eq(x):
        return cv2.equalizeHist(x)

    t_ms, mem_kb = _benchmark(cv_hist_eq, gray)
    _print_row("hist_equalize", "opencv", t_ms, mem_kb)
    rows.append(
        {
            "group": "color",
            "op": "hist_equalize",
            "impl": "opencv",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    return rows


def eval_geometric_ops(img_color: np.ndarray) -> List[Dict[str, Any]]:
    _print_header("GEOMETRIC TRANSFORMS (INTERP + AFFINE/BACKWARD/SCALE/ROTATE/SHEAR)")
    rows: List[Dict[str, Any]] = []

    # ---- Scaling (1.6x) ----
    A_scale, out_size_scale = build_scale_matrix(img_color, sx=1.6, sy=1.6)
    t_ms, mem_kb = _benchmark(warp_affine_nearest, img_color, A_scale, out_size_scale)
    _print_row("scale_1.6", "manual_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "scale_1.6",
            "impl": "manual_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    t_ms, mem_kb = _benchmark(warp_affine_bilinear, img_color, A_scale, out_size_scale)
    _print_row("scale_1.6", "manual_bilinear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "scale_1.6",
            "impl": "manual_bilinear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    def cv_resize_nn(x):
        return cv2.resize(x, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)

    def cv_resize_linear(x):
        return cv2.resize(x, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)

    t_ms, mem_kb = _benchmark(cv_resize_nn, img_color)
    _print_row("scale_1.6", "opencv_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "scale_1.6",
            "impl": "opencv_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )
    t_ms, mem_kb = _benchmark(cv_resize_linear, img_color)
    _print_row("scale_1.6", "opencv_linear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "scale_1.6",
            "impl": "opencv_linear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Rotation 30deg (warpAffine) ----
    A_rot, out_size_rot = build_rotation_matrix(img_color, angle_deg=30)
    t_ms, mem_kb = _benchmark(warp_affine_nearest, img_color, A_rot, out_size_rot)
    _print_row("rotate_30", "manual_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "rotate_30",
            "impl": "manual_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    t_ms, mem_kb = _benchmark(warp_affine_bilinear, img_color, A_rot, out_size_rot)
    _print_row("rotate_30", "manual_bilinear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "rotate_30",
            "impl": "manual_bilinear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    h_rot, w_rot = out_size_rot

    def cv_rotate_nn(x):
        return cv2.warpAffine(x, A_rot, (w_rot, h_rot), flags=cv2.INTER_NEAREST)

    def cv_rotate_linear(x):
        return cv2.warpAffine(x, A_rot, (w_rot, h_rot), flags=cv2.INTER_LINEAR)

    t_ms, mem_kb = _benchmark(cv_rotate_nn, img_color)
    _print_row("rotate_30", "opencv_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "rotate_30",
            "impl": "opencv_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )
    t_ms, mem_kb = _benchmark(cv_rotate_linear, img_color)
    _print_row("rotate_30", "opencv_linear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "rotate_30",
            "impl": "opencv_linear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    # ---- Shear (kx=0.4) ----
    A_shear, out_size_shear = build_shear_matrix(img_color, kx=0.4, ky=0.0)
    t_ms, mem_kb = _benchmark(warp_affine_nearest, img_color, A_shear, out_size_shear)
    _print_row("shear_kx0.4", "manual_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "shear_kx0.4",
            "impl": "manual_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    t_ms, mem_kb = _benchmark(warp_affine_bilinear, img_color, A_shear, out_size_shear)
    _print_row("shear_kx0.4", "manual_bilinear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "shear_kx0.4",
            "impl": "manual_bilinear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    h_sh, w_sh = out_size_shear

    def cv_shear_nn(x):
        return cv2.warpAffine(x, A_shear, (w_sh, h_sh), flags=cv2.INTER_NEAREST)

    def cv_shear_linear(x):
        return cv2.warpAffine(x, A_shear, (w_sh, h_sh), flags=cv2.INTER_LINEAR)

    t_ms, mem_kb = _benchmark(cv_shear_nn, img_color)
    _print_row("shear_kx0.4", "opencv_nn", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "shear_kx0.4",
            "impl": "opencv_nn",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )
    t_ms, mem_kb = _benchmark(cv_shear_linear, img_color)
    _print_row("shear_kx0.4", "opencv_linear", t_ms, mem_kb)
    rows.append(
        {
            "group": "geom",
            "op": "shear_kx0.4",
            "impl": "opencv_linear",
            "time_ms": t_ms,
            "mem_kb": mem_kb,
        }
    )

    return rows


def _plot_bar(
    rows: List[Dict[str, Any]],
    metric: str,
    title: str,
    out_path: str,
):
    """
    Vẽ biểu đồ cột cho 1 metric (time_ms hoặc mem_kb).
    Mỗi cột là 1 (op, impl).
    """
    if not rows:
        return

    labels = [f"{r['op']}\n{r['impl']}" for r in rows]
    values = [float(r[metric]) for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(max(8, 0.5 * len(labels)), 6))
    plt.bar(x, values, color="#4C72B0")
    plt.xticks(x, labels, rotation=45, ha="right")
    if metric == "time_ms":
        plt.ylabel("Thời gian (ms)")
    else:
        plt.ylabel("Peak memory (KB)")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    img_path = "assets/Lenna.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    print(f"Input image: {img_path}, shape={img.shape}, dtype={img.dtype}")

    color_rows = eval_color_ops(img)
    geom_rows = eval_geometric_ops(img)
    all_rows = color_rows + geom_rows

    out_dir = "outputs/evaluation"
    _plot_bar(
        all_rows,
        metric="time_ms",
        title="So sánh thời gian thực thi (manual vs OpenCV)",
        out_path=os.path.join(out_dir, "benchmark_time.png"),
    )
    _plot_bar(
        all_rows,
        metric="mem_kb",
        title="So sánh peak memory (manual vs OpenCV)",
        out_path=os.path.join(out_dir, "benchmark_memory.png"),
    )
    print(f"Đã lưu biểu đồ cột vào thư mục: {out_dir}")


if __name__ == "__main__":
    main()
