"""
Main entry point for Edge Detection project.

CLI to run traditional (classical) edge detection on a single image.
By default, runs all traditional algorithms and saves results to results/classical/.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical import (
    BasicGradient,
    ForwardDifferenceOperator,
    BackwardDifferenceOperator,
    CentralDifferenceOperator,
    RobertsOperator,
    PrewittOperator,
    SobelOperator,
    FreiChenOperator,
    Laplacian4Neighbor,
    Laplacian8Neighbor,
    LaplacianOfGaussian,
    CannyEdgeDetector,
)
from utils import load_image, save_image

# List of all available traditional edge detection detectors
ALL_DETECTORS = [
    "basic_gradient",
    "forward_diff",
    "backward_diff",
    "central_diff",
    "roberts",
    "prewitt",
    "sobel",
    "freichen",
    "laplacian4",
    "laplacian8",
    "log",
    "canny",
]


def _build_detector(name: str, **kwargs):
    """
    Build and return a classical edge detector instance by name.

    Args:
        name: Name of the detector (case-insensitive). Must be one of:
            - "basic_gradient": Basic gradient operator (fx, fy, magnitude, direction)
            - "forward_diff": Forward difference operator
            - "backward_diff": Backward difference operator
            - "central_diff": Central difference operator
            - "roberts": Roberts cross operator
            - "prewitt": Prewitt operator
            - "sobel": Sobel operator
            - "freichen": Frei-Chen operator
            - "laplacian4": Laplacian 4-neighborhood operator
            - "laplacian8": Laplacian 8-neighborhood operator
            - "log": Laplacian of Gaussian (requires sigma in kwargs)
            - "canny": Canny edge detector (requires sigma, low_threshold, high_threshold in kwargs)
        **kwargs: Additional parameters for specific detectors:
            - sigma (float): For LoG and Canny detectors (default: 1.0)
            - low_threshold (float): For Canny detector (default: 0.1)
            - high_threshold (float): For Canny detector (default: 0.2)

    Returns:
        An instance of the specified edge detector class.

    Raises:
        ValueError: If the detector name is not supported.
    """
    name = name.lower()

    if name == "basic_gradient":
        return BasicGradient()
    elif name == "forward_diff":
        return ForwardDifferenceOperator()
    elif name == "backward_diff":
        return BackwardDifferenceOperator()
    elif name == "central_diff":
        return CentralDifferenceOperator()
    elif name == "roberts":
        return RobertsOperator()
    elif name == "prewitt":
        return PrewittOperator()
    elif name == "sobel":
        return SobelOperator()
    elif name == "freichen":
        return FreiChenOperator()
    elif name == "laplacian4":
        return Laplacian4Neighbor()
    elif name == "laplacian8":
        return Laplacian8Neighbor()
    elif name == "log":
        sigma = kwargs.get("sigma", 1.0)
        return LaplacianOfGaussian(sigma=sigma)
    elif name == "canny":
        sigma = kwargs.get("sigma", 1.0)
        low_threshold = kwargs.get("low_threshold", 0.1)
        high_threshold = kwargs.get("high_threshold", 0.2)
        return CannyEdgeDetector(
            sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
        )
    else:
        raise ValueError(
            f"Unsupported detector '{name}'. " f"Available: {', '.join(ALL_DETECTORS)}"
        )


def run_traditional(
    image_path: str,
    detector_name: str,
    output_dir: str = "results/classical",
    **detector_kwargs,
):
    """
    Run a single traditional edge detection algorithm on an image and save the result.

    Args:
        image_path: Path to the input image file.
        detector_name: Name of the detector to use (must be in ALL_DETECTORS).
        output_dir: Directory where the output edge map will be saved (default: "results/classical").
        **detector_kwargs: Additional keyword arguments passed to the detector constructor
            (e.g., sigma for LoG/Canny, thresholds for Canny).

    Returns:
        None. The edge map is saved to disk at output_dir/{detector_name}.png
    """
    # Load input image
    image = load_image(image_path)

    # Build detector instance
    detector = _build_detector(detector_name, **detector_kwargs)

    # Apply edge detection
    edge_map = detector(image)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save edge map with detector name as filename
    output_path = os.path.join(output_dir, f"{detector_name}.png")
    save_image(edge_map, output_path)
    print(f"Saved {detector_name} edge map to {output_path}")


def run_all_traditional(
    image_path: str,
    output_dir: str = "results/classical",
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
):
    """
    Run all available traditional edge detection algorithms on a single image.

    This function iterates through all detectors in ALL_DETECTORS, applies each one
    to the input image, and saves the results to separate files in the output directory.
    Each result file is named after its detector (e.g., "sobel.png", "canny.png").

    Args:
        image_path: Path to the input image file.
        output_dir: Directory where all edge maps will be saved (default: "results/classical").
        sigma: Sigma parameter for LoG and Canny detectors (default: 1.0).
        low_threshold: Low threshold for Canny detector (default: 0.1).
        high_threshold: High threshold for Canny detector (default: 0.2).

    Returns:
        None. All edge maps are saved to disk in the output directory.
    """
    print(f"Running all traditional edge detection algorithms on {image_path}")
    print(f"Results will be saved to {output_dir}/")
    print("-" * 60)

    # Process each detector
    for detector_name in ALL_DETECTORS:
        print(f"Processing {detector_name}...", end=" ", flush=True)

        # Prepare detector-specific parameters
        detector_kwargs = {}
        if detector_name in ["log", "canny"]:
            detector_kwargs["sigma"] = sigma
        if detector_name == "canny":
            detector_kwargs["low_threshold"] = low_threshold
            detector_kwargs["high_threshold"] = high_threshold

        # Run detector and handle errors gracefully
        try:
            run_traditional(
                image_path=image_path,
                detector_name=detector_name,
                output_dir=output_dir,
                **detector_kwargs,
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("-" * 60)
    print(f"All results saved to {output_dir}/")


def parse_args():
    """
    Parse command-line arguments for the edge detection script.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - image: Path to input image
            - detector: Specific detector name (None to run all)
            - sigma: Sigma parameter for LoG/Canny
            - low_threshold: Low threshold for Canny
            - high_threshold: High threshold for Canny
            - output_dir: Output directory for results
    """
    parser = argparse.ArgumentParser(
        description="Run traditional (classical) edge detection on a single image. "
        "By default, runs all detectors. Use --detector to run a specific one."
    )
    parser.add_argument(
        "--image",
        default="./data/RGB_008.jpg",
        help="Path to input image. Default: ./data/RGB_008.jpg",
    )
    parser.add_argument(
        "--detector",
        default=None,
        choices=ALL_DETECTORS,
        help="Run a specific detector. If not specified, runs all detectors.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma parameter for LoG or Canny detector (default: 1.0).",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.1,
        help="Low threshold for Canny detector (default: 0.1).",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.2,
        help="High threshold for Canny detector (default: 0.2).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/classical",
        help="Output directory for saved edge maps (default: results/classical).",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the edge detection script.

    Parses command-line arguments and executes either a single detector or all detectors
    based on user input. Validates input image existence before processing.

    Raises:
        FileNotFoundError: If the specified input image does not exist.
    """
    args = parse_args()

    # Validate input image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(
            f"Input image not found at {args.image}. Provide --image or place default at source/data/RGB_008.jpg"
        )

    # If a specific detector is specified, run only that detector
    if args.detector:
        # Prepare detector-specific parameters
        detector_kwargs = {}
        if args.detector in ["log", "canny"]:
            detector_kwargs["sigma"] = args.sigma
        if args.detector == "canny":
            detector_kwargs["low_threshold"] = args.low_threshold
            detector_kwargs["high_threshold"] = args.high_threshold

        run_traditional(
            image_path=args.image,
            detector_name=args.detector,
            output_dir=args.output_dir,
            **detector_kwargs,
        )
    else:
        # Default: run all detectors
        run_all_traditional(
            image_path=args.image,
            output_dir=args.output_dir,
            sigma=args.sigma,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
        )


if __name__ == "__main__":
    main()
