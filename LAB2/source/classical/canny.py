"""
Canny Edge Detector implementation.

Includes:
- Gaussian Smoothing
- Gradient Magnitude and Direction
- Non-Maximum Suppression
- Hysteresis Thresholding
"""

import numpy as np
from scipy import ndimage
from typing import Tuple
from .base import BaseEdgeDetector


class CannyEdgeDetector(BaseEdgeDetector):
    """
    Canny Edge Detector.

    Multi-stage algorithm:
    1. Gaussian smoothing
    2. Compute gradient magnitude and direction
    3. Non-maximum suppression
    4. Hysteresis thresholding
    """

    def __init__(
        self,
        sigma: float = 1.0,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
    ):
        """
        Initialize Canny detector.

        Args:
            sigma: Standard deviation for Gaussian smoothing
            low_threshold: Low threshold for hysteresis (ratio of max gradient)
            high_threshold: High threshold for hysteresis (ratio of max gradient)
        """
        super().__init__("Canny")
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Sobel kernels for gradient computation
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    def _gaussian_smooth(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing.

        Args:
            image: Input image

        Returns:
            Smoothed image
        """
        return ndimage.gaussian_filter(image, sigma=self.sigma)

    def _compute_gradient(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient magnitude and direction.

        Args:
            image: Input image

        Returns:
            Tuple of (magnitude, direction, gx, gy)
        """
        padded = np.pad(image, 1, mode="edge")
        h, w = image.shape

        gx = np.zeros((h, w), dtype=np.float64)
        gy = np.zeros((h, w), dtype=np.float64)

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                gx[i - 1, j - 1] = np.sum(patch * self.sobel_x)
                gy[i - 1, j - 1] = np.sum(patch * self.sobel_y)

        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)  # In radians, range [-pi, pi]

        return magnitude, direction, gx, gy

    def _non_maximum_suppression(
        self, magnitude: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """
        Apply non-maximum suppression.

        Args:
            magnitude: Gradient magnitude
            direction: Gradient direction

        Returns:
            Suppressed edge map
        """
        h, w = magnitude.shape
        suppressed = np.zeros((h, w), dtype=np.float64)

        # Convert direction to [0, 180] degrees and quantize to 4 directions
        angle = np.degrees(direction) % 180

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                mag = magnitude[i, j]

                # Determine which neighbors to check based on gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    # Horizontal: compare with left and right
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif 22.5 <= angle[i, j] < 67.5:
                    # Diagonal (top-right to bottom-left)
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif 67.5 <= angle[i, j] < 112.5:
                    # Vertical: compare with top and bottom
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:  # 112.5 <= angle < 157.5
                    # Diagonal (top-left to bottom-right)
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                # Keep pixel if it's a local maximum
                if mag >= max(neighbors):
                    suppressed[i, j] = mag

        return suppressed

    def _hysteresis_thresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply hysteresis thresholding.

        Args:
            image: Non-maximum suppressed image

        Returns:
            Binary edge map
        """
        max_val = image.max()
        low_thresh = max_val * self.low_threshold
        high_thresh = max_val * self.high_threshold

        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.uint8)

        # Strong edges (above high threshold)
        strong_edges = image >= high_thresh
        edge_map[strong_edges] = 255

        # Weak edges (between low and high threshold)
        weak_edges = (image >= low_thresh) & (image < high_thresh)

        # Connect weak edges to strong edges
        # Use 8-connectivity
        changed = True
        while changed:
            changed = False
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if weak_edges[i, j] and edge_map[i, j] == 0:
                        # Check 8 neighbors
                        neighbors = edge_map[i - 1 : i + 2, j - 1 : j + 2]
                        if np.any(neighbors == 255):
                            edge_map[i, j] = 255
                            changed = True

        return edge_map

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection.

        Args:
            image: Input grayscale image

        Returns:
            Binary edge map
        """
        # Step 1: Gaussian smoothing
        smoothed = self._gaussian_smooth(image)

        # Step 2: Compute gradient
        magnitude, direction, _, _ = self._compute_gradient(smoothed)

        # Step 3: Non-maximum suppression
        suppressed = self._non_maximum_suppression(magnitude, direction)

        # Step 4: Hysteresis thresholding
        edge_map = self._hysteresis_thresholding(suppressed)

        return edge_map.astype(np.float64)
