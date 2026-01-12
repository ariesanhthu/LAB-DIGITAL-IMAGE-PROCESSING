from __future__ import annotations

import numpy as np

from .base import BaseSmoothing


class MeanSmoothing(BaseSmoothing):
    """Mean/average filter implementation."""

    @staticmethod
    def apply(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Run mean filter with given odd kernel size.

        Args:
            image: Input image.
            kernel_size: Odd kernel size.

        Returns:
            np.ndarray: Mean-filtered image.

        Raises:
            ValueError: If kernel_size is even.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        gray = BaseSmoothing._ensure_gray(image)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
            kernel_size * kernel_size
        )
        out = BaseSmoothing._convolve(gray, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)
