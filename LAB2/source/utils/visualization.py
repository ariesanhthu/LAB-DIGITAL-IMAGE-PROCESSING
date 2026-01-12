"""
Visualization utilities for edge detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from PIL import Image


def visualize_edge_detection(
    original: np.ndarray,
    edge_map: np.ndarray,
    title: str = "Edge Detection",
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Visualize original image and edge map side by side.

    Args:
        original: Original image
        edge_map: Detected edge map
        title: Figure title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(edge_map, cmap="gray")
    axes[1].set_title("Edge Map")
    axes[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def compare_edge_detectors(
    original: np.ndarray,
    edge_maps: List[np.ndarray],
    detector_names: List[str],
    figsize: Tuple[int, int] = (15, 10),
):
    """
    Compare multiple edge detection results.

    Args:
        original: Original image
        edge_maps: List of edge maps from different detectors
        detector_names: List of detector names
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_detectors = len(edge_maps)
    n_cols = 3
    n_rows = (n_detectors + 2) // n_cols  # +1 for original, +1 for rounding

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Show original
    axes[0].imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show edge maps
    for i, (edge_map, name) in enumerate(zip(edge_maps, detector_names), 1):
        if i < len(axes):
            axes[i].imshow(edge_map, cmap="gray")
            axes[i].set_title(name)
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(edge_maps) + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig
