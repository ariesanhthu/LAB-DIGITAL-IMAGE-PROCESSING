from __future__ import annotations

import numpy as np
from .base import BaseSmoothing


class ConditionalSmoothing(BaseSmoothing):
    """
    Conditional smoothing filters including:

    1) Range-based conditional filters (textbook DIP style):
       - A pixel is considered valid if its intensity lies within [min_value, max_value].
       - Used in classical DIP exercises for illustrating conditional averaging.
       - Includes two variants:
           a) range_impulse: only replaces center pixels outside the range.
           b) range_smooth: smooths all pixels using neighbors within the range.

    2) Difference-based conditional filter (modern practical version):
       - A pixel (i,j) contributes if |f(i,j) - f(x,y)| <= threshold.
       - More adaptive and suitable for natural images and Gaussian noise.
       - Often behaves similarly to a lightweight bilateral/edge-preserving filter.

    This class exposes clear naming to avoid confusion between textbook
    and modern conditional filters.
    """

    # ==========================================================
    # 1) RANGE-BASED CONDITIONAL FILTER (TEXTBOOK STYLE)
    # ==========================================================
    @staticmethod
    def _range_conditional(
        image: np.ndarray,
        kernel_size: int,
        min_value: float,
        max_value: float,
        *,
        replace_outside_only: bool,
    ) -> np.ndarray:
        """
        Core operator for range-based conditional filtering.

        A neighbor pixel is selected if:
            min_value <= f(i,j) <= max_value

        This matches the classical DIP definition where "valid" pixels
        are considered to lie inside a global intensity interval.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale or color image.
        kernel_size : int
            Odd-sized window defining the local neighborhood.
        min_value : float
            Lower bound of accepted intensity range.
        max_value : float
            Upper bound of accepted intensity range.
        replace_outside_only : bool
            If True:
                Only center pixels lying OUTSIDE the interval [min,max]
                will be replaced by the mean of valid neighbors.
                (Used for textbook impulse-noise removal.)
            If False:
                All pixels are replaced by the valid-neighbor mean
                whenever such neighbors exist.
                (Used for smoothing while ignoring invalid intensities.)

        Returns
        -------
        np.ndarray
            Filtered grayscale image clipped to [0,255].
        """

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        gray = BaseSmoothing._ensure_gray(image).astype(np.float32)
        pad = kernel_size // 2
        padded = BaseSmoothing._pad(gray, pad)
        h, w = gray.shape
        out = gray.copy()

        for y in range(h):
            for x in range(w):
                center = gray[y, x]
                region = padded[y : y + kernel_size, x : x + kernel_size]

                # Condition: valid-intensity range [min_value, max_value]
                mask = (region >= min_value) & (region <= max_value)

                if not mask.any():
                    # No valid neighbors exist → keep original
                    out[y, x] = center
                    continue

                mean_valid = float(region[mask].mean())

                if replace_outside_only:
                    # Replace center only if outside [min,max]
                    if (center < min_value) or (center > max_value):
                        out[y, x] = mean_valid
                    else:
                        out[y, x] = center
                else:
                    # Always use valid-neighbor mean
                    out[y, x] = mean_valid

        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def range_impulse(image: np.ndarray, kernel_size: int, min_value: float, max_value: float):
        """
        Range-based conditional filter specialized for impulse (salt-and-pepper) noise.

        This algorithm follows the classical DIP rule:
            - If the center pixel lies outside [min_value, max_value],
              it is replaced by the mean of in-range neighbors.
            - Otherwise, it is preserved.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        kernel_size : int
            Odd kernel size.
        min_value : float
            Lower bound for valid intensity.
        max_value : float
            Upper bound for valid intensity.

        Returns
        -------
        np.ndarray
            Denoised output image.
        """
        return ConditionalSmoothing._range_conditional(
            image=image,
            kernel_size=kernel_size,
            min_value=min_value,
            max_value=max_value,
            replace_outside_only=True,
        )

    @staticmethod
    def range_smooth(
        image: np.ndarray,
        kernel_size: int,
        min_value: float,
        max_value: float,
    ):
        """
        Range-based smoothing filter.

        Unlike range_impulse(), this variant replaces *all* pixels
        (when valid neighbors exist) using the mean of neighbors within
        the interval [min_value, max_value].

        Useful when the image contains intensities that should be ignored
        globally (e.g., clipped values, unstable sensor spikes).

        Returns
        -------
        np.ndarray
        """
        return ConditionalSmoothing._range_conditional(
            image=image,
            kernel_size=kernel_size,
            min_value=min_value,
            max_value=max_value,
            replace_outside_only=False,
        )

    # ==========================================================
    # 2) DIFFERENCE-BASED CONDITIONAL FILTER (MODERN STYLE)
    # ==========================================================
    @staticmethod
    def diff_filter(
        image: np.ndarray,
        kernel_size: int,
        threshold: float,
    ) -> np.ndarray:
        """
        Modern conditional filtering using local pixel similarity.

        A neighbor contributes if:
            |f(i,j) - f(x,y)| <= threshold

        This method is adaptive (depends on center pixel), and is
        significantly more suitable for natural images and Gaussian noise
        than the textbook range-based version. It resembles a simplified
        bilateral filter: edges tend to be preserved because intensity
        discontinuities exceed the threshold.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale or color image.
        kernel_size : int
            Odd-sized local window.
        threshold : float
            Maximum allowed absolute deviation from the center pixel.

        Returns
        -------
        np.ndarray
            Edge-preserving smoothed output.
        """

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        gray = BaseSmoothing._ensure_gray(image).astype(np.float32)
        pad = kernel_size // 2
        padded = BaseSmoothing._pad(gray, pad)
        h, w = gray.shape
        out = gray.copy()

        for y in range(h):
            for x in range(w):
                center = gray[y, x]
                region = padded[y : y + kernel_size, x : x + kernel_size]

                diff = np.abs(region - center)
                mask = diff <= threshold

                if mask.any():
                    out[y, x] = float(region[mask].mean())
                else:
                    out[y, x] = center

        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def conditional_modern(image: np.ndarray, kernel_size: int, threshold: float):
        """
        Alias for the difference-based conditional filter.

        Provided purely for naming clarity inside experiments and reports.
        """
        return ConditionalSmoothing.diff_filter(image, kernel_size, threshold)
