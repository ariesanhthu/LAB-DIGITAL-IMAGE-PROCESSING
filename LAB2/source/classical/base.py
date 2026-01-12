"""
Base class for classical edge detection algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, Optional


class BaseEdgeDetector(ABC):
    """
    Abstract base class for all classical edge detection algorithms.

    All edge detection algorithms should inherit from this class and
    implement the detect() method.
    """

    def __init__(self, name: str):
        """
        Initialize the edge detector.

        Args:
            name: Name of the edge detection algorithm
        """
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the input image.

        Args:
            image: Input grayscale image (2D numpy array)

        Returns:
            Edge map (2D numpy array with same shape as input)
        """
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image (convert to grayscale if needed).

        Args:
            image: Input image (can be grayscale or RGB)

        Returns:
            Grayscale image (2D numpy array)
        """
        if len(image.shape) == 3:
            # Convert RGB to grayscale using standard weights
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        elif len(image.shape) != 2:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image.astype(np.float64)

    def normalize(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Normalize edge map to [0, 255] range.

        Args:
            edge_map: Raw edge map

        Returns:
            Normalized edge map (uint8)
        """
        if edge_map.max() == edge_map.min():
            return np.zeros_like(edge_map, dtype=np.uint8)

        normalized = (
            (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min()) * 255
        )
        return normalized.astype(np.uint8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Make the detector callable.

        Args:
            image: Input image

        Returns:
            Edge map
        """
        processed_image = self.preprocess(image)
        edge_map = self.detect(processed_image)
        return self.normalize(edge_map)
