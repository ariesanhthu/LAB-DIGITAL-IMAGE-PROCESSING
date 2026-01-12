from __future__ import annotations

import numpy as np

from .base import BaseSmoothing


class AdaptiveSmoothing(BaseSmoothing):
    """Adaptive and advanced smoothing filters."""

    @staticmethod
    def gradient_weighted(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Gradient/gray-level weighted averaging (bilateral-style weights).

        Args:
            image: Input image.
            kernel_size: Odd window size.

        Returns:
            np.ndarray: Filtered image.

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
        eps = BaseSmoothing.epsilon
        for y in range(h):
            for x in range(w):
                region = padded[y : y + kernel_size, x : x + kernel_size]
                center = gray[y, x]
                diff = np.abs(region - center)
                delta = 1.0 / (diff + eps)
                weights = delta / np.sum(delta)
                out[y, x] = np.sum(region * weights)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def gradient_weighted_impulse(
        image: np.ndarray, kernel_size: int, threshold: float
    ) -> np.ndarray:
        """Impulse-robust gradient weighting (handles salt & pepper).

        Uses local median to detect outlier centers; when center is outlier,
        re-center weights around the median and ignore far pixels (> threshold).

        Args:
            image: Input image.
            kernel_size: Odd window size.
            threshold: Intensity threshold to classify outliers.

        Returns:
            np.ndarray: Filtered image.

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
        eps = BaseSmoothing.epsilon

        for y in range(h):
            for x in range(w):
                region = padded[y : y + kernel_size, x : x + kernel_size]
                center = gray[y, x]
                med = np.median(region)

                if abs(center - med) > threshold:
                    ref = med
                    diff = np.abs(region - ref)
                    mask = diff <= threshold
                    if np.any(mask):
                        delta = np.zeros_like(region)
                        delta[mask] = 1.0 / (diff[mask] + eps)
                    else:
                        out[y, x] = med
                        continue
                else:
                    diff = np.abs(region - center)
                    delta = 1.0 / (diff + eps)

                weights = delta / np.sum(delta)
                out[y, x] = np.sum(region * weights)

        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def rotating_mask(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Rotating mask averaging (variance-based orientation selection).

        Args:
            image: Input image.
            kernel_size: Must be 3 for current implementation.

        Returns:
            np.ndarray: Filtered image.

        Raises:
            ValueError: If kernel_size is not 3.
        """
        if kernel_size != 3:
            raise ValueError("Rotating mask currently supports kernel_size=3")
        gray = BaseSmoothing._ensure_gray(image).astype(np.float32)
        pad = 1
        padded = BaseSmoothing._pad(gray, pad)
        h, w = gray.shape
        out = np.zeros_like(gray, dtype=np.float32)
        masks = [
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        ]
        for y in range(h):
            for x in range(w):
                region = padded[y : y + 3, x : x + 3]
                best_var = None
                best_mean = None
                for m in masks:
                    vals = region[m == 1]
                    if vals.size == 0:
                        continue
                    var = np.var(vals)
                    mean = np.mean(vals)
                    if best_var is None or var < best_var:
                        best_var = var
                        best_mean = mean
                out[y, x] = best_mean if best_mean is not None else np.mean(region)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def mmse(image: np.ndarray, kernel_size: int, noise_variance: float) -> np.ndarray:
        """MMSE filter using local mean/variance and known noise variance.

        Args:
            image: Input image.
            kernel_size: Odd window size.
            noise_variance: Estimated noise variance.

        Returns:
            np.ndarray: MMSE filtered image.

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
        eps = BaseSmoothing.epsilon
        for y in range(h):
            for x in range(w):
                region = padded[y : y + kernel_size, x : x + kernel_size]
                local_mean = np.mean(region)
                local_var = np.var(region)
                center = gray[y, x]
                if local_var < noise_variance + eps:
                    out[y, x] = local_mean
                else:
                    out[y, x] = center - (noise_variance / (local_var + eps)) * (
                        center - local_mean
                    )
        return np.clip(out, 0, 255).astype(np.uint8)
