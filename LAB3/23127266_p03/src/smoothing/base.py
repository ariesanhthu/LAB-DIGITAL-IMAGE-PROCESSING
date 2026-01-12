from __future__ import annotations

import numpy as np


class BaseSmoothing:
    """Base utilities for spatial smoothing filters."""

    epsilon: float = 1e-5

    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        """Convert input to grayscale uint8.

        Args:
            img: Input image (gray or color).

        Returns:
            np.ndarray: Grayscale uint8 image.
        """
        if img.ndim == 2:
            return img.astype(np.uint8)
        return np.mean(img, axis=2).astype(np.uint8)

    @staticmethod
    def _pad(image: np.ndarray, pad: int) -> np.ndarray:
        """Reflect-pad image to avoid border artifacts.

        Args:
            image: Grayscale image.
            pad: Padding size on each side.

        Returns:
            np.ndarray: Padded image.
        """
        return np.pad(image, pad, mode="reflect")

    @staticmethod
    def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Manual convolution with odd-sized square kernel.

        Args:
            image: Grayscale image.
            kernel: Odd-sized square kernel.

        Returns:
            np.ndarray: Convolved image (float32).

        Raises:
            ValueError: If kernel size is even.
        """
        k = kernel.shape[0]
        if k % 2 == 0:
            raise ValueError("kernel must be odd-sized")
        pad = k // 2
        padded = BaseSmoothing._pad(image, pad)
        h, w = image.shape
        out = np.zeros_like(image, dtype=np.float32)
        flipped = np.flipud(np.fliplr(kernel)).astype(np.float32)
        for y in range(h):
            for x in range(w):
                region = padded[y : y + k, x : x + k]
                out[y, x] = np.sum(region * flipped)
        return out
