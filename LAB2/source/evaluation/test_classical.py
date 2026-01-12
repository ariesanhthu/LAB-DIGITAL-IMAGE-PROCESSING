"""
Test và Evaluation cho các Classical Edge Detection Algorithms.

Script này test tất cả classical algorithms trên BIPED test set
và so sánh với ground truth.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical import (
    RobertsOperator,
    PrewittOperator,
    SobelOperator,
    FreiChenOperator,
    Laplacian4Neighbor,
    Laplacian8Neighbor,
    LaplacianOfGaussian,
    CannyEdgeDetector,
)
from utils import load_image, save_image
from utils.visualization import compare_edge_detectors
import matplotlib.pyplot as plt


def calculate_metrics(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tính các metrics: Precision, Recall, F1, IoU.

    Args:
        pred: Prediction array (0-255 uint8)
        target: Target array (0-255 uint8)
        threshold: Threshold để binarize (0-255)

    Returns:
        Dictionary chứa các metrics
    """
    # Normalize về 0-1
    if pred.max() > 1.0:
        pred_norm = pred.astype(np.float32) / 255.0
    else:
        pred_norm = pred.astype(np.float32)

    if target.max() > 1.0:
        target_norm = target.astype(np.float32) / 255.0
    else:
        target_norm = target.astype(np.float32)

    # Binarize với threshold (0-255 -> 0-1)
    threshold_norm = threshold / 255.0 if threshold > 1.0 else threshold
    pred_binary = (pred_norm > threshold_norm).astype(np.uint8)
    target_binary = (target_norm > 0.5).astype(np.uint8)  # GT đã là binary

    # Tính TP, FP, FN, TN
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def load_test_pairs(
    list_file: str, image_dir: str, label_dir: str
) -> List[Tuple[str, str]]:
    """
    Load danh sách test image và label pairs.

    Args:
        list_file: File list (format: "img_path label_path")
        image_dir: Root directory của images
        label_dir: Root directory của labels

    Returns:
        List of (image_path, label_path) tuples
    """
    pairs = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                img_rel_path = parts[0]
                label_rel_path = parts[1]
                img_path = os.path.join(image_dir, img_rel_path)
                label_path = os.path.join(label_dir, label_rel_path)
                if os.path.exists(img_path) and os.path.exists(label_path):
                    pairs.append((img_path, label_path))
    return pairs


def test_classical_algorithms(
    test_pairs: List[Tuple[str, str]],
    output_dir: str = "results/classical",
    threshold: float = 128,
    max_samples: int = None,
    save_images: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Test tất cả classical algorithms.

    Args:
        test_pairs: List of (image_path, label_path) tuples
        output_dir: Directory để lưu kết quả
        threshold: Threshold để binarize edge maps (0-255)
        max_samples: Số lượng samples tối đa để test (None = tất cả)
        save_images: Có lưu ảnh kết quả không

    Returns:
        Dictionary chứa metrics cho từng algorithm
    """
    # Initialize detectors
    detectors = {
        "Roberts": RobertsOperator(),
        "Prewitt": PrewittOperator(),
        "Sobel": SobelOperator(),
        "FreiChen": FreiChenOperator(),
        "Laplacian4": Laplacian4Neighbor(),
        "Laplacian8": Laplacian8Neighbor(),
        "LoG": LaplacianOfGaussian(sigma=1.0),
        "Canny": CannyEdgeDetector(sigma=1.0, low_threshold=0.1, high_threshold=0.2),
    }

    # Initialize metrics storage
    all_metrics = {
        name: {"precision": [], "recall": [], "f1": [], "iou": []}
        for name in detectors.keys()
    }

    # Limit samples if specified
    if max_samples:
        test_pairs = test_pairs[:max_samples]

    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Test on each image
    print(f"\nTesting {len(test_pairs)} images...")
    for idx, (img_path, label_path) in enumerate(tqdm(test_pairs, desc="Processing")):
        # Load image and ground truth
        image = load_image(img_path)
        gt = np.array(Image.open(label_path).convert("L"))

        # Run each detector
        edge_maps = {}
        for name, detector in detectors.items():
            edge_map = detector(image)
            edge_maps[name] = edge_map

            # Calculate metrics
            metrics = calculate_metrics(edge_map, gt, threshold=threshold)
            all_metrics[name]["precision"].append(metrics["precision"])
            all_metrics[name]["recall"].append(metrics["recall"])
            all_metrics[name]["f1"].append(metrics["f1"])
            all_metrics[name]["iou"].append(metrics["iou"])

        # Save sample images (first few)
        if save_images and idx < 5:
            sample_output_dir = os.path.join(output_dir, "images", f"sample_{idx+1}")
            os.makedirs(sample_output_dir, exist_ok=True)

            # Save original and GT
            save_image(image, os.path.join(sample_output_dir, "original.jpg"))
            save_image(gt, os.path.join(sample_output_dir, "ground_truth.png"))

            # Save edge maps
            for name, edge_map in edge_maps.items():
                save_image(edge_map, os.path.join(sample_output_dir, f"{name}.png"))

            # Create comparison visualization
            edge_map_list = [edge_maps[name] for name in detectors.keys()]
            fig = compare_edge_detectors(
                image, edge_map_list, list(detectors.keys()), figsize=(20, 12)
            )
            fig.savefig(
                os.path.join(sample_output_dir, "comparison.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Calculate average metrics
    avg_metrics = {}
    for name in detectors.keys():
        avg_metrics[name] = {
            "precision": np.mean(all_metrics[name]["precision"]),
            "recall": np.mean(all_metrics[name]["recall"]),
            "f1": np.mean(all_metrics[name]["f1"]),
            "iou": np.mean(all_metrics[name]["iou"]),
        }

    return avg_metrics


def print_results_table(results: Dict[str, Dict[str, float]]):
    """
    In bảng kết quả dạng table.

    Args:
        results: Dictionary chứa metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - CLASSICAL EDGE DETECTION ALGORITHMS")
    print("=" * 80)
    print(
        f"{'Algorithm':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'IoU':<12}"
    )
    print("-" * 80)

    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)

    for name, metrics in sorted_results:
        print(
            f"{name:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{metrics['iou']:<12.4f}"
        )

    print("=" * 80)


def save_results_json(results: Dict[str, Dict[str, float]], output_path: str):
    """
    Lưu kết quả dạng JSON.

    Args:
        results: Dictionary chứa metrics
        output_path: Path để lưu file JSON
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def plot_metrics_comparison(results: Dict[str, Dict[str, float]], output_path: str):
    """
    Vẽ biểu đồ so sánh metrics.

    Args:
        results: Dictionary chứa metrics
        output_path: Path để lưu biểu đồ
    """
    algorithms = list(results.keys())
    metrics_names = ["precision", "recall", "f1", "iou"]
    metrics_values = {
        metric: [results[alg][metric] for alg in algorithms] for metric in metrics_names
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = metrics_values[metric]
        bars = ax.bar(algorithms, values, alpha=0.7)
        ax.set_title(metric.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics comparison plot saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test and evaluate classical edge detection algorithms"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/BIPED/edges",
        help="Root directory of BIPED dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/classical",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=128,
        help="Threshold for binarization (0-255)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (None = all)",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save sample result images",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Don't generate comparison plots",
    )

    args = parser.parse_args()

    # Convert relative path to absolute if needed
    # If dataset_root is relative, resolve from project root (parent of source/)
    if not os.path.isabs(args.dataset_root):
        # Get project root (parent directory of source/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        dataset_root = os.path.join(project_root, args.dataset_root)
    else:
        dataset_root = args.dataset_root

    # Dataset paths
    test_image_dir = os.path.join(dataset_root, "imgs/test")
    test_label_dir = os.path.join(dataset_root, "edge_maps/test")
    test_list_file = os.path.join(dataset_root, "test_rgb.lst")

    # Check if files exist
    if not os.path.exists(test_list_file):
        print(f"Error: Test list file not found: {test_list_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for dataset at: {dataset_root}")
        print(f"\nPlease check:")
        print(f"  1. Dataset path is correct: {args.dataset_root}")
        print(f"  2. File exists at: {test_list_file}")
        if os.path.exists(dataset_root):
            print(f"  3. Dataset root exists, listing contents...")
            try:
                print(f"     Files in dataset root: {os.listdir(dataset_root)[:10]}")
            except:
                pass
        return

    # Load test pairs
    print("Loading test dataset...")
    test_pairs = load_test_pairs(test_list_file, test_image_dir, test_label_dir)
    print(f"Found {len(test_pairs)} test images")

    if len(test_pairs) == 0:
        print("Error: No test images found!")
        return

    # Test algorithms
    results = test_classical_algorithms(
        test_pairs,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_samples=args.max_samples,
        save_images=args.save_images,
    )

    # Print results
    print_results_table(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_json(results, os.path.join(args.output_dir, "results.json"))

    # Plot comparison
    if not args.no_plot:
        plot_metrics_comparison(
            results, os.path.join(args.output_dir, "metrics_comparison.png")
        )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
