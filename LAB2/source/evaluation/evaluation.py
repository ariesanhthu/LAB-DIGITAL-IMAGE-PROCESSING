import os
import time
from typing import Callable, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .classical import (
    CannyEdgeDetector,
    SobelOperator,
    BasicGradient,
    ForwardDifferenceOperator,
    BackwardDifferenceOperator,
    CentralDifferenceOperator,
    RobertsOperator,
    PrewittOperator,
    FreiChenOperator,
    Laplacian4Neighbor,
    Laplacian8Neighbor,
    LaplacianVariant1,
    LaplacianVariant2,
    LaplacianVariant3,
    LaplacianVariant4,
)
import torch


def plot_pr_curves(
    method_curves: Dict[str, Dict[str, List[float]]],
    dataset_name: str,
    ax=None,
    save_path: str | None = None,
):
    """
    Plot Precision–Recall curves for multiple methods on a single dataset.

    Parameters
    ----------
    method_curves : dict
        Dictionary in the format:
        {
            "Method A": {"precision": [...], "recall": [...]},
            "Method B": {"precision": [...], "recall": [...]},
            ...
        }
        Each method name maps to its precision and recall lists.

    dataset_name : str
        Name of the dataset (e.g., "BIPED-test").

    ax : matplotlib.axes.Axes or None
        Optional. Axes on which to plot; if None, a new figure and axes are created.

    save_path : str or None
        Optional. If provided, the plot will be saved to this path (PNG).
    """
    created_fig = False
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    for method_name, data in method_curves.items():
        recall = np.asarray(data["recall"], dtype=float)
        precision = np.asarray(data["precision"], dtype=float)

        # Đảm bảo sort theo recall tăng dần
        order = np.argsort(recall)
        recall = recall[order]
        precision = precision[order]

        ax.plot(recall, precision, label=method_name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curves on {dataset_name}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left")

    if created_fig:
        # Save figure if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    return ax


def _load_biped_pairs(
    root_dir: str,
    split: str = "test",
    list_filename: str = "test_rgb.lst",
) -> List[Tuple[str, str]]:
    """
    Đọc file *.lst của BIPED để lấy (đường dẫn ảnh, đường dẫn groundtruth).

    Trả về list các path tuyệt đối.
    """
    list_path = os.path.join(root_dir, list_filename)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Cannot find list file: {list_path}")

    img_root = os.path.join(root_dir, "imgs", split)
    gt_root = os.path.join(root_dir, "edge_maps", split)

    pairs: List[Tuple[str, str]] = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, gt_rel = line.split()
            img_path = os.path.join(img_root, img_rel)
            gt_path = os.path.join(gt_root, gt_rel)
            if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                continue
            pairs.append((img_path, gt_path))
    return pairs


def _compute_pr_curve_for_method(
    pairs: List[Tuple[str, str]],
    predict_func: Callable[[np.ndarray], np.ndarray],
    num_thresholds: int = 21,
    method_name: str | None = None,
) -> Dict[str, List[float]]:
    """
    Tính precision–recall curve cho một phương pháp trên tập ảnh.

    - predict_func: nhận ảnh RGB (np.ndarray HxWx3, 0–255) → edge map (np.ndarray HxW, 0–255).
    """
    thresholds = np.linspace(0, 255, num_thresholds)
    tp = np.zeros_like(thresholds, dtype=float)
    fp = np.zeros_like(thresholds, dtype=float)
    fn = np.zeros_like(thresholds, dtype=float)

    total_imgs = len(pairs)
    for idx, (img_path, gt_path) in enumerate(pairs):
        if method_name is not None and idx % 5 == 0:
            # In mỗi ảnh (verbose)
            print(
                f"[{method_name}] processing image {idx + 1}/{total_imgs}: {os.path.basename(img_path)}"
            )
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        img_arr = np.array(img)
        gt_arr = np.array(gt)

        # Groundtruth: binarize > 0
        gt_bin = gt_arr > 0

        pred = predict_func(img_arr)  # expect 0–255
        if pred.ndim == 3:
            pred = pred[..., 0]

        for i, t in enumerate(thresholds):
            pred_bin = pred >= t

            tp[i] += np.logical_and(pred_bin, gt_bin).sum()
            fp[i] += np.logical_and(pred_bin, ~gt_bin).sum()
            fn[i] += np.logical_and(~pred_bin, gt_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }


def _compute_metrics_for_method(
    pairs: List[Tuple[str, str]],
    predict_func: Callable[[np.ndarray], np.ndarray],
    threshold: float = 127.5,
    method_name: str | None = None,
) -> Dict[str, float]:
    """
    Tính các metrics (F1, Precision, Recall, IoU, Time) cho một phương pháp.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        List các cặp (img_path, gt_path).
    predict_func : Callable
        Hàm predict nhận ảnh RGB (np.ndarray HxWx3, 0–255) → edge map (np.ndarray HxW, 0–255).
    threshold : float
        Ngưỡng để binarize prediction (default: 127.5).
    method_name : str | None
        Tên phương pháp để in log.

    Returns
    -------
    Dict[str, float]
        Dictionary chứa: "f1", "precision", "recall", "iou", "time_ms".
    """
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_intersection = 0.0
    total_union = 0.0
    total_time_ms = 0.0

    total_imgs = len(pairs)
    for idx, (img_path, gt_path) in enumerate(pairs):
        if method_name is not None:
            print(
                f"[{method_name}] processing image {idx + 1}/{total_imgs}: {os.path.basename(img_path)}"
            )

        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        img_arr = np.array(img)
        gt_arr = np.array(gt)

        # Groundtruth: binarize > 0
        gt_bin = gt_arr > 0

        # Đo thời gian predict
        start_time = time.perf_counter()
        pred = predict_func(img_arr)  # expect 0–255
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        total_time_ms += elapsed_ms

        if pred.ndim == 3:
            pred = pred[..., 0]

        # Binarize prediction
        pred_bin = pred >= threshold

        # Tính TP, FP, FN
        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, ~gt_bin).sum()
        fn = np.logical_and(~pred_bin, gt_bin).sum()

        total_tp += float(tp)
        total_fp += float(fp)
        total_fn += float(fn)

        # Tính IoU
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        total_intersection += float(intersection)
        total_union += float(union)

    # Tính metrics tổng hợp
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = total_intersection / (total_union + 1e-8)
    avg_time_ms = total_time_ms / total_imgs

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "time_ms": avg_time_ms,
    }


def evaluate_metrics_table(
    biped_root: str = os.path.join("dataset", "BIPED", "edges"),
    max_images: int | None = None,
    threshold: float = 127.5,
) -> Dict[str, Dict[str, float]]:
    """
    Tính metrics (F1, Precision, Recall, IoU, Time) cho tất cả các phương pháp.

    Parameters
    ----------
    biped_root : str
        Đường dẫn đến thư mục BIPED.
    max_images : int | None
        Số lượng ảnh tối đa để test (None = tất cả).
    threshold : float
        Ngưỡng để binarize prediction (default: 127.5).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary với format:
        {
            "MethodName": {
                "f1": ...,
                "precision": ...,
                "recall": ...,
                "iou": ...,
                "time_ms": ...
            },
            ...
        }
    """
    print(f"[INFO] Loading BIPED pairs from: {biped_root}")
    pairs = _load_biped_pairs(biped_root, split="test", list_filename="test_rgb.lst")

    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
        print(
            f"[INFO] Using only first {len(pairs)} image/label pairs "
            f"out of total for quick evaluation."
        )
    else:
        print(f"[INFO] Loaded {len(pairs)} image/label pairs for evaluation.")

    # ---- Classical methods ----
    # Gradient-based
    basic_grad = BasicGradient()
    fwd_diff = ForwardDifferenceOperator()
    bwd_diff = BackwardDifferenceOperator()
    central_diff = CentralDifferenceOperator()
    roberts = RobertsOperator()
    prewitt = PrewittOperator()
    sobel = SobelOperator()
    frei_chen = FreiChenOperator()

    # Laplacian-based
    lap4 = Laplacian4Neighbor()
    lap8 = Laplacian8Neighbor()
    lap_v1 = LaplacianVariant1()
    lap_v2 = LaplacianVariant2()
    lap_v3 = LaplacianVariant3()
    lap_v4 = LaplacianVariant4()

    # Canny
    canny = CannyEdgeDetector()

    def mk_predict(detector):
        def _predict(img_arr: np.ndarray) -> np.ndarray:
            return detector(img_arr)

        return _predict

    metrics_table: Dict[str, Dict[str, float]] = {}

    # Đánh giá các phương pháp
    methods = [
        ("BasicGradient", mk_predict(basic_grad)),
        ("ForwardDiff", mk_predict(fwd_diff)),
        ("BackwardDiff", mk_predict(bwd_diff)),
        ("CentralDiff", mk_predict(central_diff)),
        ("Roberts", mk_predict(roberts)),
        ("Prewitt", mk_predict(prewitt)),
        ("Sobel", mk_predict(sobel)),
        ("FreiChen", mk_predict(frei_chen)),
        ("Laplacian4", mk_predict(lap4)),
        ("Laplacian8", mk_predict(lap8)),
        ("LapVar1", mk_predict(lap_v1)),
        ("LapVar2", mk_predict(lap_v2)),
        ("LapVar3", mk_predict(lap_v3)),
        ("LapVar4", mk_predict(lap_v4)),
        ("Canny", mk_predict(canny)),
    ]

    for method_name, predict_func in methods:
        print(f"[INFO] Evaluating {method_name}...")
        metrics_table[method_name] = _compute_metrics_for_method(
            pairs, predict_func, threshold=threshold, method_name=method_name
        )

    return metrics_table


def print_metrics_table(metrics_table: Dict[str, Dict[str, float]]):
    """
    In bảng kết quả metrics dạng text table.

    Parameters
    ----------
    metrics_table : Dict[str, Dict[str, float]]
        Dictionary kết quả từ evaluate_metrics_table.
    """
    print("\n" + "=" * 100)
    print("METRICS TABLE")
    print("=" * 100)
    print(
        f"{'Method':<20} {'F1':<10} {'Precision':<12} {'Recall':<12} {'IoU':<12} {'Time (ms)':<12}"
    )
    print("-" * 100)

    for method_name, metrics in sorted(metrics_table.items()):
        f1 = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        iou = metrics["iou"]
        time_ms = metrics["time_ms"]

        print(
            f"{method_name:<20} {f1:<10.4f} {precision:<12.4f} {recall:<12.4f} "
            f"{iou:<12.4f} {time_ms:<12.2f}"
        )

    print("=" * 100 + "\n")


def evaluate_classical_and_deep_on_biped(
    biped_root: str = os.path.join("dataset", "BIPED", "edges"),
    max_images: int | None = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Chạy evaluation (Precision–Recall) trên BIPED-test cho:
    - Nhiều thuật toán classical (Gradient-based, Laplacian, Canny, ...)

    Trả về:
        {
            "Canny": {"precision": [...], "recall": [...]},
            "Sobel": {...},
            "Prewitt": {...},
            ...
        }
    """
    print(f"[INFO] Loading BIPED pairs from: {biped_root}")
    pairs = _load_biped_pairs(biped_root, split="test", list_filename="test_rgb.lst")

    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
        print(
            f"[INFO] Using only first {len(pairs)} image/label pairs "
            f"out of total for quick evaluation."
        )
    else:
        print(f"[INFO] Loaded {len(pairs)} image/label pairs for evaluation.")

    # ---- Classical methods ----
    # Gradient-based
    basic_grad = BasicGradient()
    fwd_diff = ForwardDifferenceOperator()
    bwd_diff = BackwardDifferenceOperator()
    central_diff = CentralDifferenceOperator()
    roberts = RobertsOperator()
    prewitt = PrewittOperator()
    sobel = SobelOperator()
    frei_chen = FreiChenOperator()

    # Laplacian-based
    lap4 = Laplacian4Neighbor()
    lap8 = Laplacian8Neighbor()
    lap_v1 = LaplacianVariant1()
    lap_v2 = LaplacianVariant2()
    lap_v3 = LaplacianVariant3()
    lap_v4 = LaplacianVariant4()

    # Canny
    canny = CannyEdgeDetector()

    def mk_predict(detector):
        def _predict(img_arr: np.ndarray) -> np.ndarray:
            return detector(img_arr)

        return _predict

    method_curves: Dict[str, Dict[str, List[float]]] = {}

    # Đánh giá một số phương pháp tiêu biểu
    print("[INFO] Evaluating Basic Gradient...")
    method_curves["BasicGradient"] = _compute_pr_curve_for_method(
        pairs, mk_predict(basic_grad), method_name="BasicGradient"
    )

    print("[INFO] Evaluating Forward Difference...")
    method_curves["ForwardDiff"] = _compute_pr_curve_for_method(
        pairs, mk_predict(fwd_diff), method_name="ForwardDiff"
    )

    print("[INFO] Evaluating Backward Difference...")
    method_curves["BackwardDiff"] = _compute_pr_curve_for_method(
        pairs, mk_predict(bwd_diff), method_name="BackwardDiff"
    )

    print("[INFO] Evaluating Central Difference...")
    method_curves["CentralDiff"] = _compute_pr_curve_for_method(
        pairs, mk_predict(central_diff), method_name="CentralDiff"
    )

    print("[INFO] Evaluating Roberts...")
    method_curves["Roberts"] = _compute_pr_curve_for_method(
        pairs, mk_predict(roberts), method_name="Roberts"
    )

    print("[INFO] Evaluating Prewitt...")
    method_curves["Prewitt"] = _compute_pr_curve_for_method(
        pairs, mk_predict(prewitt), method_name="Prewitt"
    )

    print("[INFO] Evaluating Sobel...")
    method_curves["Sobel"] = _compute_pr_curve_for_method(
        pairs, mk_predict(sobel), method_name="Sobel"
    )

    print("[INFO] Evaluating Frei-Chen...")
    method_curves["FreiChen"] = _compute_pr_curve_for_method(
        pairs, mk_predict(frei_chen), method_name="FreiChen"
    )

    print("[INFO] Evaluating Laplacian-4...")
    method_curves["Laplacian4"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap4), method_name="Laplacian4"
    )

    print("[INFO] Evaluating Laplacian-8...")
    method_curves["Laplacian8"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap8), method_name="Laplacian8"
    )

    print("[INFO] Evaluating Laplacian-Variant1...")
    method_curves["LapVar1"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap_v1), method_name="LapVar1"
    )

    print("[INFO] Evaluating Laplacian-Variant2...")
    method_curves["LapVar2"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap_v2), method_name="LapVar2"
    )

    print("[INFO] Evaluating Laplacian-Variant3...")
    method_curves["LapVar3"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap_v3), method_name="LapVar3"
    )

    print("[INFO] Evaluating Laplacian-Variant4...")
    method_curves["LapVar4"] = _compute_pr_curve_for_method(
        pairs, mk_predict(lap_v4), method_name="LapVar4"
    )

    print("[INFO] Evaluating Canny...")
    method_curves["Canny"] = _compute_pr_curve_for_method(
        pairs, mk_predict(canny), method_name="Canny"
    )

    return method_curves


def test_biped_evaluation():
    """
    Hàm tiện ích để gọi nhanh evaluation và vẽ PR curves.

    Cách dùng (từ root project):

        from source.evaluation import test_biped_evaluation
        test_biped_evaluation()
    """
    print("[INFO] Starting BIPED evaluation...")
    biped_root = os.path.join("dataset", "BIPED", "edges")

    # Đổi max_images thành 5 để chạy nhanh (chỉ 5 hình)
    method_curves = evaluate_classical_and_deep_on_biped(
        biped_root=biped_root,
        max_images=10,
    )

    # Đường dẫn lưu hình kết quả evaluation
    save_path = os.path.join("result", "biped_pr_curves.png")
    print(f"[INFO] Plotting and saving PR curves to: {save_path}")

    plot_pr_curves(
        method_curves,
        dataset_name="BIPED-test",
        save_path=save_path,
    )
    print("[INFO] Evaluation finished.")


def test_metrics_table():
    """
    Hàm tiện ích để tính và in bảng metrics.

    Cách dùng (từ root project):

        from source.evaluation import test_metrics_table
        test_metrics_table()
    """
    print("[INFO] Starting metrics table evaluation...")
    biped_root = os.path.join("dataset", "BIPED", "edges")

    # Test trên 5 ảnh
    metrics_table = evaluate_metrics_table(
        biped_root=biped_root,
        max_images=5,
        threshold=127.5,
    )

    # In bảng kết quả
    print_metrics_table(metrics_table)
    print("[INFO] Metrics table evaluation finished.")


if __name__ == "__main__":
    # test_biped_evaluation()
    test_metrics_table()
