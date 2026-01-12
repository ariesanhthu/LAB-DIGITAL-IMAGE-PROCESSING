"""
Laplacian edge detection operators:
- 4-neighborhood Laplacian
- 8-neighborhood Laplacian
- Laplacian mask variants
"""

import numpy as np
from .base import BaseEdgeDetector


class Laplacian4Neighbor(BaseEdgeDetector):
    """
    4-neighborhood Laplacian operator for edge detection.

    Uses only 4 direct neighbors (up, down, left, right).
    """

    def __init__(self):
        super().__init__("Laplacian-4")
        # 4-neighborhood Laplacian kernel
        self.kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 4-neighborhood Laplacian operator.

        Args:
            image: Input grayscale image

        Returns:
            Edge map (zero-crossings indicate edges)
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernel with padding
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map


class Laplacian8Neighbor(BaseEdgeDetector):
    """
    8-neighborhood Laplacian operator for edge detection.

    Uses all 8 neighbors including diagonals.
    """

    def __init__(self):
        super().__init__("Laplacian-8")
        # 8-neighborhood Laplacian kernel
        self.kernel = np.array(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 8-neighborhood Laplacian operator.

        Args:
            image: Input grayscale image

        Returns:
            Edge map (zero-crossings indicate edges)
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernel with padding
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map


class LaplacianVariant1(BaseEdgeDetector):
    """
    Laplacian Variant 1: 4-neighborhood with negative center.

    Kernel:
    [ 0 -1  0]
    [-1 -4 -1]
    [ 0 -1  0]
    """

    def __init__(self):
        super().__init__("Laplacian-Variant1")
        self.kernel = np.array([[0, -1, 0], [-1, -4, -1], [0, -1, 0]], dtype=np.float64)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian variant 1."""
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map


class LaplacianVariant2(BaseEdgeDetector):
    """
    Laplacian Variant 2: 8-neighborhood with negative center.

    Kernel:
    [-1 -1 -1]
    [-1 -8 -1]
    [-1 -1 -1]
    """

    def __init__(self):
        super().__init__("Laplacian-Variant2")
        self.kernel = np.array(
            [[-1, -1, -1], [-1, -8, -1], [-1, -1, -1]], dtype=np.float64
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian variant 2."""
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map


class LaplacianVariant3(BaseEdgeDetector):
    """
    Laplacian Variant 3: Enhanced diagonal response.

    Kernel:
    [-2 -1 -2]
    [-1  12 -1]
    [-2 -1 -2]
    """

    def __init__(self):
        super().__init__("Laplacian-Variant3")
        self.kernel = np.array(
            [[-2, -1, -2], [-1, 12, -1], [-2, -1, -2]], dtype=np.float64
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian variant 3."""
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map


class LaplacianVariant4(BaseEdgeDetector):
    """
    Laplacian Variant 4: Isotropic variant.

    Kernel:
    [-1 -2 -1]
    [-2 12 -2]
    [-1 -2 -1]
    """

    def __init__(self):
        super().__init__("Laplacian-Variant4")
        self.kernel = np.array(
            [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]], dtype=np.float64
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian variant 4."""
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                edge_map[i - 1, j - 1] = np.abs(np.sum(patch * self.kernel))

        return edge_map
