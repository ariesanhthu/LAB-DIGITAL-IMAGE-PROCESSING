"""
Laplacian of Gaussian (LoG) edge detection.
"""

import numpy as np
from scipy import ndimage
from .base import BaseEdgeDetector


class LaplacianOfGaussian(BaseEdgeDetector):
    """
    Laplacian of Gaussian (LoG) edge detector.

    Applies Gaussian smoothing first, then Laplacian operator.
    Detects edges by finding zero-crossings in the LoG response.
    """

    def __init__(self, sigma: float = 1.0, kernel_size: int = None):
        """
        Initialize LoG detector.

        Args:
            sigma: Standard deviation of Gaussian kernel
            kernel_size: Size of the kernel (if None, computed from sigma)
        """
        super().__init__("LoG")
        self.sigma = sigma

        if kernel_size is None:
            # Default kernel size: 6*sigma rounded to nearest odd number
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

        self.kernel_size = kernel_size
        self.log_kernel = self._create_log_kernel()

    def _create_log_kernel(self) -> np.ndarray:
        """
        Create Laplacian of Gaussian kernel.

        Returns:
            LoG kernel
        """
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        center = self.kernel_size // 2

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x = i - center
                y = j - center
                r_squared = x**2 + y**2
                # LoG formula: (r^2 - 2*sigma^2) / (sigma^4) * exp(-r^2/(2*sigma^2))
                kernel[i, j] = (
                    (r_squared - 2 * self.sigma**2)
                    / (self.sigma**4)
                    * np.exp(-r_squared / (2 * self.sigma**2))
                )

        # Normalize to sum to zero
        kernel = kernel - np.mean(kernel)
        return kernel

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply LoG operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge map (absolute value of LoG response)
        """
        # Apply LoG kernel
        edge_map = ndimage.convolve(image, self.log_kernel, mode="constant")

        # Return absolute value
        return np.abs(edge_map)

    def detect_zero_crossings(
        self, image: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """
        Detect edges using zero-crossing method.

        Args:
            image: Input grayscale image
            threshold: Minimum magnitude for zero-crossing detection

        Returns:
            Binary edge map
        """
        # Apply LoG
        log_response = ndimage.convolve(image, self.log_kernel, mode="constant")

        # Find zero-crossings
        h, w = log_response.shape
        edge_map = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check for zero-crossing in 4 directions
                neighbors = [
                    log_response[i - 1, j],
                    log_response[i + 1, j],
                    log_response[i, j - 1],
                    log_response[i, j + 1],
                ]

                center = log_response[i, j]

                # Zero-crossing: sign change and magnitude > threshold
                for neighbor in neighbors:
                    if (center * neighbor < 0) and (
                        np.abs(center - neighbor) > threshold
                    ):
                        edge_map[i, j] = 255
                        break

        return edge_map
