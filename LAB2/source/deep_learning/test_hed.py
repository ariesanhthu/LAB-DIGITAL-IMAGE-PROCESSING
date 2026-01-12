"""
Test HED pretrained model with images from data folder.
Uses OpenCV DNN to load Caffe model (compatible with hed_pretrained_bsds.caffemodel).
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_hed_caffe(prototxt_path: str = None, caffemodel_path: str = None):
    """
    Load HED model using OpenCV DNN from Caffe format.

    Args:
        prototxt_path: Path to deploy.prototxt file
        caffemodel_path: Path to hed_pretrained_bsds.caffemodel file

    Returns:
        OpenCV DNN network object
    """
    # Default paths - check UNet_edge_detection folder first
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    unet_dir = os.path.join(base_dir, "UNet_edge_detection")

    if prototxt_path is None:
        # Try UNet_edge_detection first
        prototxt_candidate = os.path.join(unet_dir, "deploy.prototxt.txt")
        if os.path.exists(prototxt_candidate):
            prototxt_path = prototxt_candidate
        else:
            raise FileNotFoundError(
                f"Could not find deploy.prototxt.txt. Checked: {prototxt_candidate}"
            )

    if caffemodel_path is None:
        # Try UNet_edge_detection first
        caffemodel_candidate = os.path.join(unet_dir, "hed_pretrained_bsds.caffemodel")
        if os.path.exists(caffemodel_candidate):
            caffemodel_path = caffemodel_candidate
        else:
            raise FileNotFoundError(
                f"Could not find hed_pretrained_bsds.caffemodel. Checked: {caffemodel_candidate}"
            )

    # Load network using OpenCV DNN
    net = cv2.dnn.readNet(prototxt_path, caffemodel_path)
    print(f"Loaded HED model from:")
    print(f"  Prototxt: {prototxt_path}")
    print(f"  Caffemodel: {caffemodel_path}")

    return net


def predict_hed_opencv(net, image, threshold=0.5):
    """
    Predict edges using HED model loaded with OpenCV DNN.

    Args:
        net: OpenCV DNN network
        image: PIL Image or numpy array (RGB)
        threshold: Threshold for binarization

    Returns:
        Binary edge map as numpy array (uint8, 0-255)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # HED expects BGR format and specific preprocessing
    # Convert RGB to BGR
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # Get image dimensions
    (H, W) = img_bgr.shape[:2]

    # Create blob from image
    # HED model expects mean subtraction: (104.00698793, 116.66876762, 122.67891434)
    mean_values = (104.00698793, 116.66876762, 122.67891434)
    blob = cv2.dnn.blobFromImage(
        img_bgr,
        scalefactor=1.0,
        size=(W, H),
        mean=mean_values,
        swapRB=False,  # Already BGR
        crop=False,
    )

    # Set input and forward pass
    # HED model output layer is "sigmoid-fuse" (final fused output)
    net.setInput(blob)
    output = net.forward("sigmoid-fuse")

    # Output shape is typically (1, 1, H, W) or (1, H, W)
    if len(output.shape) == 4:
        edge_map = output[0, 0]  # Remove batch and channel dims
    elif len(output.shape) == 3:
        edge_map = output[0]  # Remove batch dim
    else:
        edge_map = output

    # Output is already sigmoid (0-1 range), just apply threshold
    edge_map = (edge_map > threshold).astype(np.uint8) * 255

    return edge_map


def test_hed_on_data(
    data_dir: str = None,
    output_dir: str = None,
    prototxt_path: str = None,
    caffemodel_path: str = None,
    threshold: float = 0.5,
):
    """
    Test HED pretrained model on images in data folder.

    Args:
        data_dir: Directory containing test images (None to auto-detect)
        output_dir: Directory to save results (None to auto-detect)
        prototxt_path: Path to deploy.prototxt.txt (None to auto-detect)
        caffemodel_path: Path to hed_pretrained_bsds.caffemodel (None to auto-detect)
        threshold: Threshold for edge binarization
    """
    # Auto-detect paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if data_dir is None:
        data_dir = os.path.join(base_dir, "source", "data")

    if output_dir is None:
        output_dir = os.path.join(base_dir, "source", "results", "deeplearning")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading HED pretrained model...")
    try:
        net = load_hed_caffe(
            prototxt_path=prototxt_path, caffemodel_path=caffemodel_path
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure hed_pretrained_bsds.caffemodel and deploy.prototxt.txt")
        print("are available in UNet_edge_detection folder or provide paths manually.")
        return

    # Get all image files in data directory
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f
        for f in os.listdir(data_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {data_dir}")
        return

    print(f"Found {len(image_files)} image(s) to process")

    # Process each image
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        print(f"\nProcessing: {img_file}")

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
            print(f"  Image size: {image.size}")
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue

        # Predict edges
        edge_map = predict_hed_opencv(net, image, threshold=threshold)

        # Save result
        output_name = f"hed_{os.path.splitext(img_file)[0]}.png"
        output_path = os.path.join(output_dir, output_name)

        edge_image = Image.fromarray(edge_map, mode="L")
        edge_image.save(output_path)
        print(f"  Saved result to: {output_path}")

        # Optionally create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(edge_map, cmap="gray")
        axes[1].set_title("HED Edge Detection")
        axes[1].axis("off")

        vis_name = f"hed_vis_{os.path.splitext(img_file)[0]}.png"
        vis_path = os.path.join(output_dir, vis_name)
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved visualization to: {vis_path}")

    print("\nTesting completed!")


if __name__ == "__main__":
    # Test HED model
    test_hed_on_data(
        data_dir=None,  # Auto-detect
        output_dir=None,  # Auto-detect
        prototxt_path=None,  # Auto-detect from UNet_edge_detection
        caffemodel_path=None,  # Auto-detect from UNet_edge_detection
        threshold=0.5,
    )
