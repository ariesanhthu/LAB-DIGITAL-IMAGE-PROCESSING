from __future__ import annotations

import os
import platform
from pathlib import Path

import cv2
import numpy as np


class FileIOHandle:
    """Handle filesystem layout for outputs and common save helpers."""

    def __init__(self, out_dir: Path) -> None:
        """Initialize output directories.

        Args:
            out_dir: Root output directory.
        """
        self.paths = {
            "root": out_dir,
            "pre": out_dir / "preprocessingImage",
            "result": out_dir / "result",
            "compare": out_dir / "compare",
        }
        for path in self.paths.values():
            ensure_dir(path)

    def save_single_image(
        self,
        image: np.ndarray,
        noise_tag: str,
        filter_name: str,
        label: str,
    ) -> Path:
        """Save one filtered image into structured result directories.

        Args:
            image: Image to save.
            noise_tag: Noise key ('sp' or 'gaussian').
            filter_name: Name of filter (e.g., 'mean', 'median').
            label: Identifier such as kernel size.

        Returns:
            Path: Path to the saved image.
        """
        out_dir = self.paths["result"] / noise_tag / filter_name
        ensure_dir(out_dir)
        out_path = out_dir / f"{filter_name}_{noise_tag}_{label}.png"
        cv2.imwrite(str(out_path), image)
        return out_path


try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if missing.

    Args:
        path: Target directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def add_noise(
    image: np.ndarray, mode: str = "gaussian", amount: float = 10.0
) -> np.ndarray:
    """
    Add synthetic noise for filter testing.

    Args:
        image: Input image (gray or color).
        mode: 'gaussian' or 's&p'.
        amount: Std for gaussian; ratio for salt & pepper.

    Returns:
        np.ndarray: Noisy image.

    Raises:
        ValueError: If mode is invalid.
    """
    img = image.astype(np.float32).copy()
    if mode == "gaussian":
        noise = np.random.normal(0, amount, size=img.shape).astype(np.float32)
        noisy = img + noise
    elif mode in {"s&p", "salt_pepper"}:
        noisy = img.copy()
        prob = float(np.clip(amount, 0.0, 1.0))
        mask = np.random.rand(*noisy.shape[:2])
        salt = mask < (prob / 2)
        pepper = (mask >= (prob / 2)) & (mask < prob)
        if noisy.ndim == 3:
            noisy[salt] = 255
            noisy[pepper] = 0
        else:
            noisy[salt] = 255
            noisy[pepper] = 0
    else:
        raise ValueError("mode phải là 'gaussian' hoặc 's&p'")
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure 3-channel BGR for visualization.

    Args:
        img: Grayscale or color image.

    Returns:
        np.ndarray: BGR image.
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def make_tile(img: np.ndarray, label: str, tile_size=(256, 256)) -> np.ndarray:
    """Resize and add header label for grid visualization.

    Args:
        img: Image to tile.
        label: Text label.
        tile_size: (w, h) of tile.

    Returns:
        np.ndarray: Labeled tile.
    """
    tile = cv2.resize(_to_bgr(img), tile_size)
    header_h = 30
    header = np.zeros((header_h, tile.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (tile.shape[1], header_h), (35, 35, 35), -1)

    # Use PIL for text rendering if available (better font support)
    if HAS_PIL and label:
        text_img = _render_text_pil(
            label.upper()[:32], tile.shape[1], header_h, 12, (35, 35, 35)
        )
        header = text_img
    else:
        cv2.putText(
            header,
            label.upper()[:32],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, tile])


def save_grid(
    images: dict[str, np.ndarray],
    out_path: Path,
    cols: int = 3,
    tile_size=(256, 256),
    pad: int = 8,
    bg_color=(15, 15, 15),
) -> None:
    """Save a labeled grid of images with uniform tile size and padding.

    Args:
        images: Mapping name -> image.
        out_path: Output path.
        cols: Number of columns.
        tile_size: Tile size (w, h).
        pad: Padding (px) between tiles.
        bg_color: Background color tuple (B, G, R).
    """
    if not images:
        return
    tiles = [make_tile(img, name, tile_size) for name, img in images.items()]
    if not tiles:
        return
    h, w, c = tiles[0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)
    blank[:, :] = bg_color
    while len(tiles) % cols != 0:
        tiles.append(blank.copy())

    padded_rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = []
        for tile in tiles[i : i + cols]:
            tile_pad = cv2.copyMakeBorder(
                tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color
            )
            row_tiles.append(tile_pad)
        padded_rows.append(cv2.hconcat(row_tiles))

    grid = padded_rows[0] if len(padded_rows) == 1 else cv2.vconcat(padded_rows)
    cv2.imwrite(str(out_path), grid)


def _render_text_pil(
    text: str, width: int, height: int, font_size: int = 16, bg_color=(15, 15, 15)
) -> np.ndarray:
    """Render text using PIL for better font support.

    Args:
        text: Text to render.
        width: Image width.
        height: Image height.
        font_size: Font size.
        bg_color: Background color (R, G, B).

    Returns:
        np.ndarray: BGR image with rendered text.
    """
    import platform
    import os

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try multiple font paths based on OS
    font_paths = []
    system = platform.system()

    if system == "Windows":
        # Windows font paths
        windows_fonts_dir = os.path.join(
            os.environ.get("WINDIR", "C:/Windows"), "Fonts"
        )
        font_paths.extend(
            [
                os.path.join(windows_fonts_dir, "arial.ttf"),
                os.path.join(windows_fonts_dir, "Arial.ttf"),
                os.path.join(windows_fonts_dir, "arialbd.ttf"),  # Arial Bold
                os.path.join(windows_fonts_dir, "calibri.ttf"),
                os.path.join(windows_fonts_dir, "Calibri.ttf"),
                os.path.join(windows_fonts_dir, "segoeui.ttf"),  # Segoe UI
                os.path.join(windows_fonts_dir, "tahoma.ttf"),
                os.path.join(windows_fonts_dir, "Tahoma.ttf"),
            ]
        )
    elif system == "Linux":
        # Linux font paths
        font_paths.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            ]
        )
    elif system == "Darwin":  # macOS
        # macOS font paths
        font_paths.extend(
            [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        )

    # Add generic fallbacks
    font_paths.extend(
        [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
        ]
    )

    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                # Test if font can render the text (especially Unicode characters)
                try:
                    test_bbox = draw.textbbox((0, 0), text, font=font)
                    if test_bbox[2] > test_bbox[0]:  # Valid bounding box
                        break
                except Exception:
                    continue
        except (OSError, IOError, Exception):
            continue

    # If no font found or font doesn't support the text, try default
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        # Fallback: estimate size
        text_width = len(text) * font_size // 2
        text_height = font_size

    # Center text
    x = max(0, (width - text_width) // 2)
    y = max(0, (height - text_height) // 2 - (bbox[1] if "bbox" in locals() else 0))

    try:
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    except Exception:
        # If rendering fails, try with ASCII fallback (replace Unicode × with x)
        text_ascii = text.replace("×", "x")
        draw.text((x, y), text_ascii, fill=(255, 255, 255), font=font)

    # Convert PIL to OpenCV format (RGB -> BGR)
    img_np = np.array(img)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def save_table_grid(
    rows_data: list[tuple[str, np.ndarray] | tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    header_labels: tuple[str, str, str] = ("Kernel size", "Salt & Pepper", "Gaussian"),
    tile_size=(256, 256),
    pad: int = 10,
    bg_color=(20, 20, 20),
    single_image: bool = False,
    header_bg=(60, 60, 60),
    border_color=(100, 100, 100),
    cols: int | None = None,
) -> None:
    """Save a table-style grid with header row and kernel size column.

    Args:
        rows_data: List of tuples. Each tuple is either (label, image)
            or (label, image1, image2). If only one image is provided,
            the function will render a single tile for that entry.
        out_path: Output path.
        header_labels: (col1_label, col2_label, col3_label).
        tile_size: Tile size (w, h).
        pad: Padding (px) between tiles.
        bg_color: Background color tuple (B, G, R).
        header_bg: Header background color tuple (B, G, R).
        border_color: Border color tuple (B, G, R).
        cols: If provided, arrange as grid instead of table (ignores header_labels).
    """
    if not rows_data:
        return

    # Grid mode: arrange items in grid layout
    if cols is not None:
        # Show single image if explicitly requested, only one image provided,
        # or both provided images are identical.
        if rows_data:
            first = rows_data[0]
            if len(first) == 2:
                single_image = True
            elif len(first) >= 3 and np.array_equal(first[1], first[2]):
                single_image = True
        _save_comparison_grid(
            rows_data, out_path, cols, tile_size, pad, bg_color, single_image
        )
        return

    # Table mode: fixed 3-column layout
    header_h = 45
    tile_w, tile_h = tile_size
    kernel_col_w = 140
    border_w = 2

    # Create header row with borders
    header_row = []
    # Kernel size header
    kernel_header = np.zeros((header_h, kernel_col_w, 3), dtype=np.uint8)
    kernel_header[:, :] = header_bg
    if HAS_PIL:
        text_img = _render_text_pil(
            header_labels[0], kernel_col_w, header_h, 14, tuple(reversed(header_bg))
        )
        kernel_header = text_img
    else:
        cv2.putText(
            kernel_header,
            header_labels[0],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # Add border
    kernel_header = cv2.copyMakeBorder(
        kernel_header,
        border_w,
        border_w,
        border_w,
        border_w,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    header_row.append(kernel_header)

    # Salt & Pepper header
    sp_header_w = tile_w + 2 * pad
    sp_header = np.zeros((header_h, sp_header_w, 3), dtype=np.uint8)
    sp_header[:, :] = header_bg
    if HAS_PIL:
        text_img = _render_text_pil(
            header_labels[1], sp_header_w, header_h, 14, tuple(reversed(header_bg))
        )
        sp_header = text_img
    else:
        cv2.putText(
            sp_header,
            header_labels[1],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    sp_header = cv2.copyMakeBorder(
        sp_header,
        border_w,
        border_w,
        border_w,
        border_w,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    header_row.append(sp_header)

    # Gaussian header
    gauss_header_w = tile_w + 2 * pad
    gauss_header = np.zeros((header_h, gauss_header_w, 3), dtype=np.uint8)
    gauss_header[:, :] = header_bg
    if HAS_PIL:
        text_img = _render_text_pil(
            header_labels[2], gauss_header_w, header_h, 14, tuple(reversed(header_bg))
        )
        gauss_header = text_img
    else:
        cv2.putText(
            gauss_header,
            header_labels[2],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    gauss_header = cv2.copyMakeBorder(
        gauss_header,
        border_w,
        border_w,
        border_w,
        border_w,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    header_row.append(gauss_header)
    header_line = cv2.hconcat(header_row)

    # Create data rows
    data_rows = []
    row_h = tile_h + 2 * pad + 30
    for row in rows_data:
        if len(row) == 2:
            kernel_label, sp_img = row
            gauss_img = sp_img
            use_single = True
        else:
            kernel_label, sp_img, gauss_img = row
            use_single = False
        row_tiles = []

        # Kernel size column (text only)
        kernel_cell = np.zeros((row_h, kernel_col_w, 3), dtype=np.uint8)
        kernel_cell[:, :] = bg_color
        if HAS_PIL:
            text_img = _render_text_pil(
                kernel_label, kernel_col_w, row_h, 16, tuple(reversed(bg_color))
            )
            kernel_cell = text_img
        else:
            cv2.putText(
                kernel_cell,
                kernel_label,
                (10, row_h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        kernel_cell = cv2.copyMakeBorder(
            kernel_cell,
            border_w,
            border_w,
            border_w,
            border_w,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        row_tiles.append(kernel_cell)

        # Salt & Pepper image tile
        sp_tile = make_tile(sp_img, "", tile_size)
        sp_padded = cv2.copyMakeBorder(
            sp_tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color
        )
        sp_padded = cv2.copyMakeBorder(
            sp_padded,
            border_w,
            border_w,
            border_w,
            border_w,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        row_tiles.append(sp_padded)

        # Gaussian image tile (optional)
        if not use_single:
            gauss_tile = make_tile(gauss_img, "", tile_size)
            gauss_padded = cv2.copyMakeBorder(
                gauss_tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color
            )
            gauss_padded = cv2.copyMakeBorder(
                gauss_padded,
                border_w,
                border_w,
                border_w,
                border_w,
                cv2.BORDER_CONSTANT,
                value=border_color,
            )
            row_tiles.append(gauss_padded)

        data_rows.append(cv2.hconcat(row_tiles))

    # Combine header and data
    all_rows = [header_line] + data_rows
    table = cv2.vconcat(all_rows)
    cv2.imwrite(str(out_path), table)


def _save_comparison_grid(
    rows_data: list[tuple[str, np.ndarray] | tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    cols: int,
    tile_size=(256, 256),
    pad: int = 10,
    bg_color=(20, 20, 20),
    single_image: bool = False,
) -> None:
    """Save comparison items in a grid layout (for adaptive filters comparison).

    Args:
        rows_data: List of (method_name, sp_image, gauss_image) tuples.
        out_path: Output path.
        cols: Number of columns.
        tile_size: Tile size (w, h).
        pad: Padding (px) between tiles.
        bg_color: Background color tuple (B, G, R).
        single_image: If True, only show first image (sp_img), ignore gauss_img.
    """
    if not rows_data:
        return

    # Create tiles: each item shows method name + image(s)
    tiles = []
    tile_w, tile_h = tile_size
    border_w = 2
    border_color = (100, 100, 100)

    for row in rows_data:
        if len(row) == 2:
            method_name, sp_img = row
            gauss_img = sp_img
            use_single = True
        else:
            method_name, sp_img, gauss_img = row
            use_single = single_image
        # Create a combined tile: label header + image(s)
        header_h = 35

        # Resize images first
        sp_resized = cv2.resize(_to_bgr(sp_img), tile_size)

        if use_single:
            # Only show one image
            img_row = sp_resized
        else:
            # Combine Salt & Pepper and Gaussian side by side
            gauss_resized = cv2.resize(_to_bgr(gauss_img), tile_size)
            img_row = cv2.hconcat([sp_resized, gauss_resized])

        img_row = cv2.copyMakeBorder(
            img_row, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color
        )

        # Calculate combined width after padding (img_row width)
        combined_w = img_row.shape[1]

        # Header with method name (must match img_row width)
        header = np.zeros((header_h, combined_w, 3), dtype=np.uint8)
        header[:, :] = (60, 60, 60)
        if HAS_PIL:
            text_img = _render_text_pil(
                method_name, combined_w, header_h, 14, (60, 60, 60)
            )
            header = text_img
        else:
            cv2.putText(
                header,
                method_name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Stack header and images (now they have the same width)
        tile = cv2.vconcat([header, img_row])
        tile = cv2.copyMakeBorder(
            tile,
            border_w,
            border_w,
            border_w,
            border_w,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        tiles.append(tile)

    # Arrange in grid
    if not tiles:
        return

    h, w, c = tiles[0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)
    blank[:, :] = bg_color
    while len(tiles) % cols != 0:
        tiles.append(blank.copy())

    rows = []
    for i in range(0, len(tiles), cols):
        row = cv2.hconcat(tiles[i : i + cols])
        rows.append(row)

    grid = rows[0] if len(rows) == 1 else cv2.vconcat(rows)
    cv2.imwrite(str(out_path), grid)
