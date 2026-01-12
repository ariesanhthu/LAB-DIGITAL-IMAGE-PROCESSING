from .base import BaseSmoothing
from .mean import MeanSmoothing
from .gaussian import GaussianSmoothing
from .median import MedianSmoothing
from .adaptive import AdaptiveSmoothing
from .conditional import ConditionalSmoothing


class SpatialSmoothing(BaseSmoothing):
    """Facade providing a unified API for common spatial smoothing filters."""

    @staticmethod
    def mean(image, kernel_size):
        """Apply mean/average filter."""
        return MeanSmoothing.apply(image, kernel_size)

    @staticmethod
    def conditional(image, kernel_size, threshold):
        """Apply modern conditional filter based on local intensity difference."""
        return ConditionalSmoothing.diff_filter(image, kernel_size, threshold)

    @staticmethod
    def gaussian(image, kernel_size, sigma):
        """Apply Gaussian smoothing filter."""
        return GaussianSmoothing.apply(image, kernel_size, sigma)

    @staticmethod
    def median(image, kernel_size):
        """Apply median filter."""
        return MedianSmoothing.apply(image, kernel_size)

    @staticmethod
    def conditional_range_impulse(image, kernel_size, min_value, max_value):
        """Apply range-based conditional averaging tuned for impulse noise."""
        return ConditionalSmoothing.range_impulse(
            image, kernel_size, min_value, max_value
        )

    @staticmethod
    def conditional_range_smooth(image, kernel_size, min_value, max_value):
        """Apply range-based smoothing using only in-range neighbors."""
        return ConditionalSmoothing.range_smooth(
            image, kernel_size, min_value, max_value
        )

    @staticmethod
    def gradient_weighted(image, kernel_size):
        """Apply gradient-weighted averaging."""
        return AdaptiveSmoothing.gradient_weighted(image, kernel_size)

    @staticmethod
    def gradient_weighted_impulse(image, kernel_size, threshold):
        """Apply impulse-robust gradient weighting (salt & pepper)."""
        return AdaptiveSmoothing.gradient_weighted_impulse(
            image, kernel_size, threshold
        )

    @staticmethod
    def rotating_mask(image, kernel_size=3):
        """Apply rotating mask filter."""
        return AdaptiveSmoothing.rotating_mask(image, kernel_size)

    @staticmethod
    def mmse(image, kernel_size, noise_variance):
        """Apply MMSE filter using local statistics."""
        return AdaptiveSmoothing.mmse(image, kernel_size, noise_variance)
