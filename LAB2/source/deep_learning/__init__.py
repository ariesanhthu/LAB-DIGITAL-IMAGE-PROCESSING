"""
Deep Learning-Based Edge Detection Module

This module contains implementations for HED edge detection using OpenCV DNN.
"""

from .test_hed import load_hed_caffe, predict_hed_opencv

__all__ = [
    "load_hed_caffe",
    "predict_hed_opencv",
]
