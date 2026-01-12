from __future__ import annotations

import numpy as np

from .base import BaseSmoothing


class GaussianSmoothing(BaseSmoothing):
    """Gaussian smoothing filter."""

    @staticmethod
    def apply(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """Run Gaussian filter with odd kernel size and sigma.

        Args:
            image: Input image.
            kernel_size: Odd kernel size.
            sigma: Gaussian sigma.

        Returns:
            np.ndarray: Gaussian-filtered image.

        Raises:
            ValueError: If kernel_size is even.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        gray = BaseSmoothing._ensure_gray(image)
        k = kernel_size // 2
        y, x = np.mgrid[-k : k + 1, -k : k + 1]
        kernel = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(
            -(x**2 + y**2) / (2 * sigma**2)
        )
        kernel = kernel / np.sum(kernel)
        out = BaseSmoothing._convolve(gray, kernel.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)
