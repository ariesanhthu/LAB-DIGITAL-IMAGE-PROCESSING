from __future__ import annotations

import numpy as np

from .base import BaseSmoothing


class MedianSmoothing(BaseSmoothing):
    """Median filter implementation."""

    @staticmethod
    def apply(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Run median filter with odd window size.

        Args:
            image: Input image.
            kernel_size: Odd window size.

        Returns:
            np.ndarray: Median-filtered image.

        Raises:
            ValueError: If kernel_size is even.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        gray = BaseSmoothing._ensure_gray(image).astype(np.float32)
        pad = kernel_size // 2
        padded = BaseSmoothing._pad(gray, pad)
        h, w = gray.shape
        out = np.zeros_like(gray, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                region = padded[y : y + kernel_size, x : x + kernel_size]
                out[y, x] = np.median(region)
        return np.clip(out, 0, 255).astype(np.uint8)
