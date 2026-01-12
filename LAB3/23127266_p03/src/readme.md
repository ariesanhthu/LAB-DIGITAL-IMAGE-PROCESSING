# LAB 3 – Spatial Smoothing Filters

This project implements spatial smoothing filters for image denoising:

- **Mean/Average Filter** (3x3, 5x5, 7x7)
- **Gaussian Filter** (3x3, 5x5, 7x7)
- **Median Filter** (3x3, 5x5, 7x7)
- **Conditional (difference-based)** (3x3, 5x5, 7x7)
- **Conditional (range-based: impulse / smooth)** (3x3, 5x5, 7x7)
- **Gradient-weighted Averaging** (3x3, 5x5, 7x7)
- **Gradient-weighted Impulse-Robust** (3x3)
- **Rotating Mask Averaging** (3x3)
- **MMSE Filter** (3x3, 5x5, 7x7)

The code processes images with both **Salt & Pepper** and **Gaussian** noise, applies all filters, and generates comparison tables with OpenCV baselines.

---

## 1. Installation

### 1.1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 1.2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `numpy>=1.21.0` – Numerical operations
- `opencv-python>=4.5.0` – Image processing and I/O
- `Pillow>=9.0.0` – Enhanced text rendering (optional but recommended)

---

## 2. Project Structure

```
src/
├── main.py              # Entry point: SmoothingApp class (config + pipeline)
├── smoothing/           # Package: Base + derived filters + facade
│   ├── __init__.py      # SpatialSmoothing facade
│   ├── base.py          # BaseSmoothing helpers
│   ├── mean.py          # MeanSmoothing
│   ├── gaussian.py      # GaussianSmoothing
│   ├── median.py        # MedianSmoothing
│   ├── adaptive.py      # AdaptiveSmoothing (gradient, rotating, mmse, impulse robust)
│   └── conditional.py   # ConditionalSmoothing (diff, range_impulse, range_smooth)
├── utils.py             # Utilities: noise generation, grid/table saving
├── evaluation.py        # Evaluation script with quantitative metrics
├── testing/             # Pipeline/test runners
│   └── test_main_pipeline.py
├── requirements.txt     # Python dependencies
├── readme.md           # This file
├── assets/
│   └── Lena.jpg        # Default test image (auto-downloaded if missing)
└── outputs/            # Generated outputs
    ├── preprocessingImage/    # Original and noisy images
    │   ├── original_color.png
    │   ├── original_gray.png
    │   ├── noisy_salt_pepper.png
    │   └── noisy_gaussian.png
    ├── result/                # Singles + tables
    │   ├── sp/                # Salt & Pepper singles
    │   │   ├── mean/mean_sp_3x3.png ...
    │   │   ├── gaussian/gaussian_sp_3x3.png ...
    │   │   ├── median/median_sp_3x3.png ...
    │   │   ├── conditional_diff/conditional_diff_sp_3x3.png
    │   │   ├── conditional_range/conditional_range_sp_3x3.png
    │   │   ├── gradient_weighted/gradient_weighted_sp_3x3.png
    │   │   ├── rotating_mask/rotating_mask_sp_3x3.png
    │   │   ├── mmse/mmse_sp_5x5.png
    │   │   ├── cv_blur/cv_blur_sp_3x3.png
    │   │   ├── cv_gaussian/cv_gaussian_sp_3x3.png
    │   │   ├── cv_median/cv_median_sp_3x3.png
    │   │   └── cv_bilateral/cv_bilateral_sp_3x3.png
    │   ├── gaussian/          # Gaussian-noise singles (same layout)
    │   │   └── ...
    │   ├── table_mean_filter.png
    │   ├── table_gaussian_filter.png
    │   ├── table_median_filter.png
    │   ├── table_conditional_diff.png
    │   ├── table_conditional_range.png
    │   ├── table_gradient_weighted.png
    │   ├── table_rotating_mask.png
    │   └── table_mmse_filter.png
    ├── compare/               # Comparison tables with OpenCV
    │   ├── compare_mean_vs_blur.png
    │   ├── compare_gaussian_vs_gaussian.png
    │   ├── compare_median_vs_median.png
    │   ├── compare_adaptive_vs_baseline_salt_pepper.png
    │   ├── compare_adaptive_vs_baseline_gaussian.png
    │   ├── compare_conditional_variants.png
    │   └── compare_mmse_vs_gaussian.png
    └── eval/                  # Evaluation results (from evaluation.py)
        ├── original.png
        ├── noisy_salt_pepper.png
        ├── noisy_gaussian.png
        └── [filter_name]_salt_pepper.png / [filter_name]_gaussian.png
```

### 2.1. Module Overview

| Module          | Purpose                 | Key Classes/Functions                                                                                                                                   |
| --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`       | Application entry point | `SmoothingApp` class, CLI argument parsing                                                                                                              |
| `smoothing/`    | Filter implementations  | `SpatialSmoothing` facade; modules: mean, gaussian, median, adaptive (incl. impulse-robust gradient), conditional (diff, range_impulse, range_smooth)   |
| `utils.py`      | Helper utilities        | `FileIOHandle` class, `add_noise()`, `save_table_grid()`, `save_grid()`, `ensure_dir()`, `make_tile()`, `_render_text_pil()`, `_save_comparison_grid()` |
| `evaluation.py` | Quantitative evaluation | `mse()`, `psnr()`, `detect_edges()`, `evaluate_edge_preservation()`, `load_filtered_image()`, `format_filter_info()`, `main()`                          |
| `testing/`      | Test runners            | `test_main_pipeline.py`, `testing/test_conditional_full.py` (range vs diff sweeps)                                                                      |

---

## 3. Usage

### 3.1. Run with Default Image (Lena)

If no image path is provided, the program uses `assets/Lena.jpg` (downloads automatically if missing):

```bash
python main.py
```

**Output:**

- Processes Lena image
- Generates noisy versions (Salt & Pepper, Gaussian)
- Applies all 7 smoothing filters
- Saves result tables to `outputs/result/`
- Creates comparison tables with OpenCV baselines in `outputs/compare/`

### 3.2. Run with Custom Image

Provide your own image path:

```bash
python main.py --image path/to/your_image.jpg
```

**Example:**

```bash
python main.py --image data/camera.png
```

### 3.3. Custom Output Directory

Specify a custom output directory:

```bash
python main.py --image path/to/image.jpg --out_dir custom_outputs
```

**Example:**

```bash
python main.py --out_dir results_test
```

### 3.4. Run Evaluation Script

Run quantitative evaluation with metrics (MSE, PSNR, edge preservation):

```bash
python evaluation.py
```

**Output:**

- Processes all filters with quantitative metrics
- Generates evaluation images in `outputs/eval/`
- Prints summary tables for metrics and execution time

---

## 4. Output Structure

### 4.1. Preprocessing Images (`outputs/preprocessingImage/`)

| File                    | Description                             |
| ----------------------- | --------------------------------------- |
| `original_color.png`    | Original color image (BGR)              |
| `original_gray.png`     | Grayscale version                       |
| `noisy_salt_pepper.png` | Grayscale with Salt & Pepper noise (5%) |
| `noisy_gaussian.png`    | Grayscale with Gaussian noise (σ=12)    |

### 4.2. Result Singles & Tables (`outputs/result/`)

- Singles: saved under `outputs/result/<noise>/<filter>/...png` for every method and noise type (includes OpenCV baselines).
- Tables: aggregated views per method:

| File                          | Table Structure                                          |
| ----------------------------- | -------------------------------------------------------- |
| `table_mean_filter.png`       | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_gaussian_filter.png`   | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_median_filter.png`     | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_conditional_diff.png`  | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_conditional_range.png` | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_gradient_weighted.png` | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |
| `table_rotating_mask.png`     | Kernel size \| Salt & Pepper \| Gaussian (3x3)           |
| `table_mmse_filter.png`       | Kernel size \| Salt & Pepper \| Gaussian (3x3, 5x5, 7x7) |

### 4.3. Comparison Tables (`outputs/compare/`)

Comparison tables show custom filters vs OpenCV baselines:

| File                                           | Comparison Content                        |
| ---------------------------------------------- | ----------------------------------------- |
| `compare_mean_vs_blur.png`                     | Mean filter 3x3 vs `cv2.blur`             |
| `compare_gaussian_vs_gaussian.png`             | Gaussian filter 5x5 vs `cv2.GaussianBlur` |
| `compare_median_vs_median.png`                 | Median filter 5x5 vs `cv2.medianBlur`     |
| `compare_adaptive_vs_baseline_salt_pepper.png` | Adaptive customs vs OpenCV (S\&P)         |
| `compare_adaptive_vs_baseline_gaussian.png`    | Adaptive customs vs OpenCV (Gaussian)     |
| `compare_conditional_variants.png`             | Conditional diff vs range (2 noises)      |
| `compare_mmse_vs_gaussian.png`                 | MMSE filter 5x5 vs `cv2.GaussianBlur`     |

---

## 5. Command Line Arguments

```bash
python main.py [OPTIONS]
```

| Argument    | Type | Default           | Description                                       |
| ----------- | ---- | ----------------- | ------------------------------------------------- |
| `--image`   | Path | `assets/Lena.jpg` | Input image path (auto-downloads Lena if missing) |
| `--out_dir` | Path | `outputs`         | Output directory for all results                  |

---

## 6. Processing Pipeline

1. **Load Image**

   - Read BGR image, convert to grayscale
   - Save `original_color.png` and `original_gray.png`

2. **Generate Noise**

   - Add Salt & Pepper noise (amount=0.05) → `noisy_salt_pepper.png`
   - Add Gaussian noise (sigma=12.0) → `noisy_gaussian.png`

3. **Apply Filters**

   - Run all 7 smoothing filters on both noise types
   - Generate result tables for each method

4. **Generate OpenCV Baselines**

   - `cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.bilateralFilter`

5. **Create Comparisons**
   - Compare custom filters with OpenCV equivalents
   - Save comparison tables

---

## 7. Extending the Code

### 7.1. Adding a New Filter

1. Implement the filter method in `SpatialSmoothing` class (`smoothing.py`):

```python
@staticmethod
def your_filter(image: np.ndarray, ksize: int, **kwargs) -> np.ndarray:
    """Your filter implementation.

    Args:
        image: Input grayscale image.
        ksize: Kernel size.
        **kwargs: Additional parameters.

    Returns:
        Filtered image.
    """
    # Implementation here
    pass
```

2. Add it to the pipeline in `main.py` → `_run_pipeline()`:

```python
# Generate results
your_filter_sp = SpatialSmoothing.your_filter(noisy_sp, ksize)
your_filter_gauss = SpatialSmoothing.your_filter(noisy_gauss, ksize)
save_table_grid(
    [("3x3", your_filter_sp, your_filter_gauss)],
    dirs["result"] / "table_your_filter.png",
)
```

### 7.2. Modifying Noise Parameters

Edit noise generation in `main.py` → `_run_pipeline()`:

```python
noisy_sp = add_noise(gray, mode="s&p", amount=0.05)  # Change amount
noisy_gauss = add_noise(gray, mode="gaussian", amount=12.0)  # Change sigma
```

---

## 8. Troubleshooting

### 8.1. PIL Import Error

If you see warnings about PIL not being available:

- Install Pillow: `pip install Pillow`
- The program will fall back to OpenCV text rendering (may have font issues)

### 8.2. Image Not Found

If the default Lena image cannot be downloaded:

- Manually place `Lena.jpg` or `Lenna.jpg` in `assets/` folder
- Or provide your own image with `--image` argument

### 8.3. Output Directory Permissions

If you cannot write to the output directory:

- Check write permissions
- Use `--out_dir` to specify a different location

---

## 9. Technical Details

### 9.1. Filter Parameters

| Filter              | Kernel Sizes | Additional Parameters                              |
| ------------------- | ------------ | -------------------------------------------------- |
| Mean                | 3, 5, 7      | None                                               |
| Gaussian            | 3, 5, 7      | sigma=1.0 (S&P), sigma=12.0 (Gaussian noise)       |
| Median              | 3, 5, 7      | None                                               |
| Conditional (diff)  | 3, 5, 7      | threshold=20                                       |
| Conditional (range) | 3, 5, 7      | low=5, high=250 (S&P); low=60, high=200 (Gaussian) |
| Gradient-weighted   | 3, 5, 7      | None                                               |
| Rotating Mask       | 3            | None (kernel_size=3 only)                          |
| MMSE                | 3, 5, 7      | noise_variance=20.0                                |

### 9.2. Noise Parameters

| Noise Type    | Parameter | Value            |
| ------------- | --------- | ---------------- |
| Salt & Pepper | amount    | 0.05 (5% pixels) |
| Gaussian      | sigma     | 12.0             |

---

## 10. API Reference

### 10.1. SmoothingApp Class (`main.py`)

| Method                               | Description                                                           |
| ------------------------------------ | --------------------------------------------------------------------- |
| `__init__()`                         | Initialize app and parse arguments                                    |
| `_parse_args()`                      | Parse CLI arguments (`--image`, `--out_dir`)                          |
| `_fetch_default_lena()`              | Ensure default Lena image exists locally; download if missing         |
| `_load_image(path)`                  | Read image as BGR; raise if not found                                 |
| `_choose_image()`                    | Resolve image path: CLI provided or default Lena                      |
| `_run_pipeline(image_path, out_dir)` | Add noise, run all smoothing filters, and write outputs + comparisons |
| `run()`                              | Execute the app                                                       |

### 10.2. SpatialSmoothing Class (`smoothing.py`)

| Method                                                                | Description                                                                        |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `mean(image, kernel_size)`                                            | Mean/average filter with given odd kernel size                                     |
| `gaussian(image, kernel_size, sigma)`                                 | Gaussian smoothing with odd kernel size and sigma                                  |
| `median(image, kernel_size)`                                          | Median filter using odd window size                                                |
| `conditional(image, kernel_size, threshold)`                          | Difference-based conditional filter (modern style)                                 |
| `conditional_range_impulse(image, kernel_size, min_value, max_value)` | Range-based conditional filter for impulse noise (textbook style)                  |
| `conditional_range_smooth(image, kernel_size, min_value, max_value)`  | Range-based smoothing filter (textbook style)                                      |
| `gradient_weighted(image, kernel_size)`                               | Gradient/gray-level weighted averaging (bilateral-style weights)                   |
| `gradient_weighted_impulse(image, kernel_size, threshold)`            | Impulse-robust gradient weighting for salt & pepper noise                          |
| `rotating_mask(image, kernel_size)`                                   | Rotating mask averaging (variance-based orientation selection, kernel_size=3 only) |
| `mmse(image, kernel_size, noise_variance)`                            | MMSE filter using local mean/variance and known noise variance                     |

### 10.3. Utility Functions (`utils.py`)

| Class/Function                                                         | Description                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `FileIOHandle(out_dir)`                                                | Handle filesystem layout for outputs and common save helpers             |
| `FileIOHandle.save_single_image(image, noise_tag, filter_name, label)` | Save one filtered image into structured result directories               |
| `ensure_dir(path)`                                                     | Create directory (and parents) if missing                                |
| `add_noise(image, mode, amount)`                                       | Add synthetic noise for filter testing (gaussian or s&p)                 |
| `_to_bgr(img)`                                                         | Ensure 3-channel BGR for visualization                                   |
| `make_tile(img, label, tile_size)`                                     | Resize and add header label for grid visualization                       |
| `save_grid(images, out_path, cols, tile_size, pad, bg_color)`          | Save a labeled grid of images with uniform tile size and padding         |
| `_render_text_pil(text, width, height, font_size, bg_color)`           | Render text using PIL for better font support                            |
| `save_table_grid(rows_data, out_path, header_labels, ...)`             | Save a table-style grid with header row and kernel size column           |
| `_save_comparison_grid(rows_data, out_path, cols, ...)`                | Save comparison items in a grid layout (for adaptive filters comparison) |

### 10.4. Evaluation Functions (`evaluation.py`)

| Function                                                          | Description                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------- |
| `mse(image1, image2)`                                             | Calculate Mean Squared Error between two images         |
| `psnr(image1, image2, max_pixel)`                                 | Calculate Peak Signal-to-Noise Ratio between two images |
| `detect_edges(image, method)`                                     | Detect edges in image using Canny or Sobel              |
| `evaluate_edge_preservation(original, filtered, method)`          | Evaluate edge preservation capability                   |
| `load_filtered_image(result_dir, filter_name, noise_type, ksize)` | Load filtered image from result directory               |
| `format_filter_info(name, filter_type, **kwargs)`                 | Format filter information string                        |
| `main()`                                                          | Run evaluation for all filters                          |

---

## 11. References

- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/
- PIL/Pillow Documentation: https://pillow.readthedocs.io/
