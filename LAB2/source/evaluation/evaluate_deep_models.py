"""
Evaluate HED and U-Net models on BIPED dataset.
Outputs metrics table and visualization charts.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_learning.test_hed import load_hed_caffe, predict_hed_opencv
from typing import Callable
import torch.nn as nn
import torch.nn.functional as F


# EdgeUNet architecture from notebook (matches checkpoint)
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv (bilinear upsample)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EdgeUNet(nn.Module):
    """Small U-Net for edge detection (1-channel output) - matches checkpoint."""

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # 32
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits


def _load_biped_pairs(
    root_dir: str,
    split: str = "test",
    list_filename: str = "test_rgb.lst",
) -> List[Tuple[str, str]]:
    """
    Load BIPED dataset pairs (image, groundtruth) from list file.

    Returns list of absolute paths.
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


def _compute_metrics_for_method(
    pairs: List[Tuple[str, str]],
    predict_func: Callable[[np.ndarray], np.ndarray],
    threshold: float = 127.5,
    method_name: str = None,
) -> Dict[str, float]:
    """
    Compute metrics (F1, Precision, Recall, IoU, Time) for a method.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        List of (img_path, gt_path) pairs.
    predict_func : Callable
        Predict function: RGB image (np.ndarray HxWx3, 0-255) -> edge map (np.ndarray HxW, 0-255).
    threshold : float
        Threshold for binarization (default: 127.5).
    method_name : str | None
        Method name for logging.

    Returns
    -------
    Dict[str, float]
        Dictionary with: "f1", "precision", "recall", "iou", "time_ms".
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

        # Measure prediction time
        start_time = time.perf_counter()
        pred = predict_func(img_arr)  # expect 0-255
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        total_time_ms += elapsed_ms

        if pred.ndim == 3:
            pred = pred[..., 0]

        # Binarize prediction
        pred_bin = pred >= threshold

        # Compute TP, FP, FN
        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, ~gt_bin).sum()
        fn = np.logical_and(~pred_bin, gt_bin).sum()

        total_tp += float(tp)
        total_fp += float(fp)
        total_fn += float(fn)

        # Compute IoU
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        total_intersection += float(intersection)
        total_union += float(union)

    # Compute aggregated metrics
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


def load_hed_model(prototxt_path: str = None, caffemodel_path: str = None):
    """Load HED Caffe model."""
    # Auto-detect paths if not provided
    if prototxt_path is None or caffemodel_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        unet_dir = os.path.join(base_dir, "UNet_edge_detection")
        model_dir = os.path.join(base_dir, "source", "model")

        if prototxt_path is None:
            prototxt_path = os.path.join(unet_dir, "deploy.prototxt.txt")
            if not os.path.exists(prototxt_path):
                raise FileNotFoundError(
                    f"Could not find deploy.prototxt.txt at {prototxt_path}"
                )

        if caffemodel_path is None:
            # Try model directory first, then UNet_edge_detection
            caffemodel_path = os.path.join(model_dir, "hed_pretrained_bsds.caffemodel")
            if not os.path.exists(caffemodel_path):
                caffemodel_path = os.path.join(
                    unet_dir, "hed_pretrained_bsds.caffemodel"
                )
            if not os.path.exists(caffemodel_path):
                raise FileNotFoundError(
                    f"Could not find hed_pretrained_bsds.caffemodel"
                )

    net = load_hed_caffe(prototxt_path, caffemodel_path)

    def predict_func(image: np.ndarray) -> np.ndarray:
        """Predict function for HED."""
        img_pil = Image.fromarray(image)
        edge_map = predict_hed_opencv(net, img_pil, threshold=0.5)
        return edge_map

    return predict_func


def load_unet_model(checkpoint_path: str, device: str = None):
    """Load U-Net PyTorch model (EdgeUNet from notebook)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use EdgeUNet architecture that matches the checkpoint
    model = EdgeUNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats (based on edge-detection.ipynb)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume checkpoint is state_dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    def predict_func(image: np.ndarray) -> np.ndarray:
        """Predict function for U-Net."""
        # Convert to tensor
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            output = torch.sigmoid(logits)  # EdgeUNet outputs logits
            output = output.squeeze(0).cpu().numpy()

        if len(output.shape) == 3:
            output = output[0]

        edge_map = (output > 0.5).astype(np.uint8) * 255
        return edge_map

    return predict_func


def create_metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create pandas DataFrame from results."""
    data = []
    for model_name, metrics in results.items():
        data.append(
            {
                "Model": model_name,
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "IoU": metrics["iou"],
                "Time (ms)": metrics["time_ms"],
            }
        )

    df = pd.DataFrame(data)
    return df


def plot_metrics_comparison(results: Dict[str, Dict[str, float]], save_path: str):
    """Create bar charts comparing metrics."""
    models = list(results.keys())
    metrics_names = ["F1", "Precision", "Recall", "IoU"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        values = [results[model][metric_name.lower()] for model in models]

        bars = ax.bar(models, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Comparison")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison chart to: {save_path}")
    plt.close()


def plot_time_comparison(results: Dict[str, Dict[str, float]], save_path: str):
    """Create bar chart comparing inference time."""
    models = list(results.keys())
    times = [results[model]["time_ms"] for model in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, times, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time Comparison")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved time comparison chart to: {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "source", "model")
    biped_root = os.path.join(base_dir, "dataset", "BIPED", "edges")
    output_dir = os.path.join(base_dir, "source", "results", "deep_learning")

    # Model paths
    unet_checkpoint = os.path.join(model_dir, "biped_edge_unet_best.pth")

    # Check if files exist
    if not os.path.exists(unet_checkpoint):
        print(f"Error: U-Net checkpoint not found at {unet_checkpoint}")
        return

    # Load test images (5 images)
    print("Loading BIPED test images...")
    pairs = _load_biped_pairs(biped_root, split="test", list_filename="test_rgb.lst")
    if len(pairs) == 0:
        print(f"Error: No test images found in {biped_root}")
        return

    # Limit to 5 images
    pairs = pairs[:5]
    print(f"Evaluating on {len(pairs)} images")

    # Load models
    print("\nLoading models...")
    print("Loading HED...")
    try:
        hed_predict = load_hed_model()  # Auto-detect paths
    except FileNotFoundError as e:
        print(f"Error loading HED: {e}")
        return

    print("Loading U-Net...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    unet_predict = load_unet_model(unet_checkpoint, device)

    # Evaluate models
    print("\nEvaluating models...")
    results = {}

    print("\nEvaluating HED...")
    results["HED"] = _compute_metrics_for_method(
        pairs, hed_predict, threshold=127.5, method_name="HED"
    )

    print("\nEvaluating U-Net...")
    results["U-Net"] = _compute_metrics_for_method(
        pairs, unet_predict, threshold=127.5, method_name="U-Net"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create and save metrics table
    print("\nCreating metrics table...")
    df = create_metrics_table(results)
    print("\n" + "=" * 80)
    print("METRICS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save table to CSV
    csv_path = os.path.join(output_dir, "deep_models_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics table to: {csv_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    metrics_chart_path = os.path.join(output_dir, "deep_models_metrics_comparison.png")
    plot_metrics_comparison(results, metrics_chart_path)

    time_chart_path = os.path.join(output_dir, "deep_models_time_comparison.png")
    plot_time_comparison(results, time_chart_path)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
