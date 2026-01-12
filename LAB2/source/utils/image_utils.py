"""
Image processing utilities.
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (RGB, uint8)
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray, output_path: str):
    """
    Save an image to file.

    Args:
        image: Image as numpy array
        output_path: Path to save image
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    Image.fromarray(image).save(output_path)


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Preprocess image for edge detection.

    Args:
        image: Input image
        target_size: Optional (height, width) to resize
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Resize if needed
    if target_size:
        from PIL import Image

        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(target_size[::-1], Image.BILINEAR)  # PIL uses (W, H)
        image = np.array(image)

    # Normalize
    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


def postprocess_edge_map(
    edge_map: np.ndarray, threshold: Optional[float] = None
) -> np.ndarray:
    """
    Postprocess edge map.

    Args:
        edge_map: Raw edge map
        threshold: Optional threshold for binarization

    Returns:
        Processed edge map (uint8)
    """
    # Normalize to [0, 255]
    if edge_map.dtype != np.uint8:
        if edge_map.max() > 1.0:
            edge_map = (
                (edge_map - edge_map.min())
                / (edge_map.max() - edge_map.min() + 1e-8)
                * 255
            )
        else:
            edge_map = edge_map * 255
        edge_map = edge_map.astype(np.uint8)

    # Apply threshold if specified
    if threshold is not None:
        edge_map = (edge_map > threshold).astype(np.uint8) * 255

    return edge_map
