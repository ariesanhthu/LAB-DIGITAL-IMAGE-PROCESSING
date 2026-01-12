"""
Gradient-based edge detection operators:
- Basic Gradient (fx, fy, magnitude, direction)
- Differencing Operators (Forward, Backward, Central)
- Roberts Operator
- Prewitt Operator
- Sobel Operator
- Frei-Chen Operator
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseEdgeDetector


class BasicGradient(BaseEdgeDetector):
    """
    Basic Gradient Operator.

    Computes gradient components (fx, fy), magnitude, and direction.
    Uses simple differencing for gradient computation.
    """

    def __init__(self):
        super().__init__("Basic Gradient")
        self.fx = None
        self.fy = None
        self.magnitude = None
        self.direction = None

    def _compute_gradient(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient components.

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (fx, fy, magnitude, direction)
        """
        h, w = image.shape
        fx = np.zeros((h, w), dtype=np.float64)
        fy = np.zeros((h, w), dtype=np.float64)

        # Compute fx (horizontal gradient) - central difference
        for i in range(h):
            for j in range(1, w - 1):
                fx[i, j] = (image[i, j + 1] - image[i, j - 1]) / 2.0
            # Forward/backward difference at boundaries
            if w > 1:
                fx[i, 0] = image[i, 1] - image[i, 0]
                fx[i, w - 1] = image[i, w - 1] - image[i, w - 2]

        # Compute fy (vertical gradient) - central difference
        for j in range(w):
            for i in range(1, h - 1):
                fy[i, j] = (image[i + 1, j] - image[i - 1, j]) / 2.0
            # Forward/backward difference at boundaries
            if h > 1:
                fy[0, j] = image[1, j] - image[0, j]
                fy[h - 1, j] = image[h - 1, j] - image[h - 2, j]

        # Compute magnitude
        magnitude = np.sqrt(fx**2 + fy**2)

        # Compute direction (angle in radians)
        direction = np.arctan2(fy, fx)

        return fx, fy, magnitude, direction

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic gradient operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        fx, fy, magnitude, direction = self._compute_gradient(image)
        self.fx = fx
        self.fy = fy
        self.magnitude = magnitude
        self.direction = direction

        return magnitude

    def get_fx(self) -> Optional[np.ndarray]:
        """Get horizontal gradient component."""
        return self.fx

    def get_fy(self) -> Optional[np.ndarray]:
        """Get vertical gradient component."""
        return self.fy

    def get_magnitude(self) -> Optional[np.ndarray]:
        """Get gradient magnitude."""
        return self.magnitude

    def get_direction(self) -> Optional[np.ndarray]:
        """Get gradient direction (in radians)."""
        return self.direction


class ForwardDifferenceOperator(BaseEdgeDetector):
    """
    Forward Difference Operator for edge detection.

    Uses forward differencing: fx = f(x+1) - f(x)
    """

    def __init__(self):
        super().__init__("Forward Difference")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply forward difference operator.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        fx = np.zeros((h, w), dtype=np.float64)
        fy = np.zeros((h, w), dtype=np.float64)

        # Forward difference in x direction
        for i in range(h):
            for j in range(w - 1):
                fx[i, j] = image[i, j + 1] - image[i, j]

        # Forward difference in y direction
        for j in range(w):
            for i in range(h - 1):
                fy[i, j] = image[i + 1, j] - image[i, j]

        magnitude = np.sqrt(fx**2 + fy**2)
        return magnitude


class BackwardDifferenceOperator(BaseEdgeDetector):
    """
    Backward Difference Operator for edge detection.

    Uses backward differencing: fx = f(x) - f(x-1)
    """

    def __init__(self):
        super().__init__("Backward Difference")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply backward difference operator.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        fx = np.zeros((h, w), dtype=np.float64)
        fy = np.zeros((h, w), dtype=np.float64)

        # Backward difference in x direction
        for i in range(h):
            for j in range(1, w):
                fx[i, j] = image[i, j] - image[i, j - 1]

        # Backward difference in y direction
        for j in range(w):
            for i in range(1, h):
                fy[i, j] = image[i, j] - image[i - 1, j]

        magnitude = np.sqrt(fx**2 + fy**2)
        return magnitude


class CentralDifferenceOperator(BaseEdgeDetector):
    """
    Central Difference Operator for edge detection.

    Uses central differencing: fx = (f(x+1) - f(x-1)) / 2
    """

    def __init__(self):
        super().__init__("Central Difference")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply central difference operator.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        fx = np.zeros((h, w), dtype=np.float64)
        fy = np.zeros((h, w), dtype=np.float64)

        # Central difference in x direction
        for i in range(h):
            for j in range(1, w - 1):
                fx[i, j] = (image[i, j + 1] - image[i, j - 1]) / 2.0
            # Forward/backward at boundaries
            if w > 1:
                fx[i, 0] = image[i, 1] - image[i, 0]
                fx[i, w - 1] = image[i, w - 1] - image[i, w - 2]

        # Central difference in y direction
        for j in range(w):
            for i in range(1, h - 1):
                fy[i, j] = (image[i + 1, j] - image[i - 1, j]) / 2.0
            # Forward/backward at boundaries
            if h > 1:
                fy[0, j] = image[1, j] - image[0, j]
                fy[h - 1, j] = image[h - 1, j] - image[h - 2, j]

        magnitude = np.sqrt(fx**2 + fy**2)
        return magnitude


class RobertsOperator(BaseEdgeDetector):
    """
    Roberts Cross Operator for edge detection.

    Uses 2x2 kernels to approximate the gradient.
    """

    def __init__(self):
        super().__init__("Roberts")
        # Roberts kernels
        self.kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
        self.kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Roberts operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernels
        for i in range(h - 1):
            for j in range(w - 1):
                patch = image[i : i + 2, j : j + 2]
                gx = np.sum(patch * self.kernel_x)
                gy = np.sum(patch * self.kernel_y)
                edge_map[i, j] = np.sqrt(gx**2 + gy**2)

        return edge_map


class PrewittOperator(BaseEdgeDetector):
    """
    Prewitt Operator for edge detection.

    Uses 3x3 kernels to compute gradient in x and y directions.
    """

    def __init__(self):
        super().__init__("Prewitt")
        # Prewitt kernels
        self.kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        self.kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Prewitt operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernels with padding
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                gx = np.sum(patch * self.kernel_x)
                gy = np.sum(patch * self.kernel_y)
                edge_map[i - 1, j - 1] = np.sqrt(gx**2 + gy**2)

        return edge_map


class SobelOperator(BaseEdgeDetector):
    """
    Sobel Operator for edge detection.

    Similar to Prewitt but with weighted center row/column.
    """

    def __init__(self):
        super().__init__("Sobel")
        # Sobel kernels
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernels with padding
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                gx = np.sum(patch * self.kernel_x)
                gy = np.sum(patch * self.kernel_y)
                edge_map[i - 1, j - 1] = np.sqrt(gx**2 + gy**2)

        return edge_map


class FreiChenOperator(BaseEdgeDetector):
    """
    Frei-Chen Operator for edge detection.

    Uses 3x3 kernels with specific weights.
    """

    def __init__(self):
        super().__init__("Frei-Chen")
        # Frei-Chen kernels
        self.kernel_x = np.array(
            [[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]], dtype=np.float64
        )
        self.kernel_y = np.array(
            [[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]], dtype=np.float64
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Frei-Chen operator to detect edges.

        Args:
            image: Input grayscale image

        Returns:
            Edge magnitude map
        """
        h, w = image.shape
        edge_map = np.zeros((h, w), dtype=np.float64)

        # Apply kernels with padding
        padded = np.pad(image, 1, mode="edge")

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                patch = padded[i - 1 : i + 2, j - 1 : j + 2]
                gx = np.sum(patch * self.kernel_x)
                gy = np.sum(patch * self.kernel_y)
                edge_map[i - 1, j - 1] = np.sqrt(gx**2 + gy**2)

        return edge_map
