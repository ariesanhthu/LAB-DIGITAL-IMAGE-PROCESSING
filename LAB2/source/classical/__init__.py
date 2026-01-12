"""
Classical Edge Detection Algorithms Module

This module contains implementations of traditional edge detection algorithms:
- Gradient-based operators (Basic Gradient, Differencing Operators, Roberts, Prewitt, Sobel, Frei-Chen)
- Laplacian operators (4-neighborhood, 8-neighborhood, Laplacian mask variants)
- Laplacian of Gaussian (LoG)
- Canny Edge Detector
"""

from .base import BaseEdgeDetector
from .gradient import (
    BasicGradient,
    ForwardDifferenceOperator,
    BackwardDifferenceOperator,
    CentralDifferenceOperator,
    RobertsOperator,
    PrewittOperator,
    SobelOperator,
    FreiChenOperator,
)
from .laplacian import (
    Laplacian4Neighbor,
    Laplacian8Neighbor,
    LaplacianVariant1,
    LaplacianVariant2,
    LaplacianVariant3,
    LaplacianVariant4,
)
from .log import LaplacianOfGaussian
from .canny import CannyEdgeDetector

__all__ = [
    "BaseEdgeDetector",
    "BasicGradient",
    "ForwardDifferenceOperator",
    "BackwardDifferenceOperator",
    "CentralDifferenceOperator",
    "RobertsOperator",
    "PrewittOperator",
    "SobelOperator",
    "FreiChenOperator",
    "Laplacian4Neighbor",
    "Laplacian8Neighbor",
    "LaplacianVariant1",
    "LaplacianVariant2",
    "LaplacianVariant3",
    "LaplacianVariant4",
    "LaplacianOfGaussian",
    "CannyEdgeDetector",
]
