# LAB 01 – Basic Image Processing Pipeline (Python + OpenCV)

This project implements basic image processing algorithms for:

- **Color Transformations** (linear, non-linear, histogram-based) :contentReference[oaicite:1]{index=1}
- **Geometric Transformations** (affine, backward mapping, interpolation, scaling/rotation/translation/shear) :contentReference[oaicite:2]{index=2}
- **Image Smoothing** (mean, Gaussian, median filtering) :contentReference[oaicite:3]{index=3}

The code is organized so that an external agent (script, CLI user, or ChatGPT Agent) can:

1. Run a **full test suite** on the standard Lena image.
2. Run the **full pipeline** on a user-provided image.
3. Enter an **interactive mode** to:
   - choose an image,
   - choose a specific function (e.g., brightness, scaling, Gaussian filter),
   - customize key parameters.

---

## 1. Project Structure

```text
.
├── main.py                     # Entry point with 3 run-modes (test / batch / interactive)
├── color_transform.py          # Color transformation algorithms
├── geometric_transforms.py     # Geometric transformation algorithms
├── smoothing.py                # Smoothing (filtering) algorithms
├── assets/
│   └── Lenna.jpg               # (optional) standard test image, auto-downloaded if missing
└── outputs/
    ├── color/
    ├── geometric_transforms_demo/
    ├── smoothing/
    └── compare/
```

### 1.1. Color Transformations (`color_transform.py`)

Implements:

- **Linear mappings**

  - `brightness_adjust(img, b)`
  - `contrast_adjust(img, a)`
  - `brightness_contrast_adjust(img, a, b)`
  - `range_linear_mapping(gray, f1, f2, g1, g2)`

- **Non-linear mappings**

  - `log_mapping(gray, c)`
  - `exp_mapping(gray, c)`

- **Histogram-based mappings**

  - `histogram_equalization(gray)`
  - `histogram_specification(source, reference)`

These are used both in batch mode and in the interactive mode.

---

### 1.2. Geometric Transformations (`geometric_transforms.py`)

Separated into techniques and demos:

- **Interpolation techniques**

  - `interp_nearest(image, x, y)`
  - `interp_bilinear(image, x, y)`

- **Affine coordinate transforms**

  - `affine_forward(A_2x3, x, y)`
  - `affine_backward(A_inv_3x3, x_prime, y_prime)`

- **Backward mapping engine**

  - `backward_mapping(image, A_2x3, out_size, interpolate_func)`

- **Combined wrappers**

  - `warp_affine_nearest(image, A_2x3, out_size)`
  - `warp_affine_bilinear(image, A_2x3, out_size)`

- **Demo helpers** (used by `main.py`)

  - `run_geometric_experiments(image, out_dir)` – runs interpolation, affine, backward, and application demos (scale/rotate/shear/translate) and returns a dict of output paths.

---

### 1.3. Image Smoothing (`smoothing.py`)

Implements 3 classic filters:

- `mean_filter(img, ksize)`
- `gaussian_filter(img, ksize, sigma)`
- `median_filter(img, ksize)`

These are also wrapped in convenience functions inside `main.py`.

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy opencv-python
```

(Optionally add `scipy` if you want to experiment further.)

---

## 3. Run Modes (Agents API)

`main.py` exposes three run modes through the `--mode` argument:

```bash
python main.py --mode {test,batch,interactive} [--image PATH] [--out_dir DIR]
```

### 3.1. Mode 1 – `test`

**Purpose:** quick self-test of the whole pipeline.

- If `assets/Lenna.jpg` exists, it is used.
- Otherwise, Lena is automatically downloaded from the canonical URL.
- Then **all algorithms** are executed:

  - Color transforms (linear, non-linear, histogram-based)
  - Geometric transforms (scale, rotate, translate, shear, plus technique demos)
  - Smoothing filters

**Example:**

```bash
python main.py --mode test
python main.py --mode test --out_dir outputs_test
```

This is the ideal mode for automated grading or for an Agent that just wants “run everything on a known image and show results”.

---

### 3.2. Mode 2 – `batch`

**Purpose:** run the full pipeline on a user image.

The image can be provided:

1. As CLI argument: `--image path/to/image.jpg`
2. Or interactively from the command line if `--image` is missing.

**Example:**

```bash
# Using CLI argument
python main.py --mode batch --image data/camera.png

# Let the program ask for the path
python main.py --mode batch
```

In both cases, the program:

- loads the chosen image,
- saves it as `original.png`,
- runs all color transforms, geometric demos and smoothing filters,
- writes their outputs into subfolders of `--out_dir`.

---

### 3.3. Mode 3 – `interactive`

**Purpose:** allow the user (or an Agent) to:

- choose an image,
- pick **one specific function**,
- optionally customize parameters,
- run it and save the output under a chosen name.

The interactive menu looks like:

1. Select image:

   - Use Lena (download if needed), **or**
   - Provide a custom image path.

2. Main menu:

   - `1` – Color Transformations
   - `2` – Geometric Transformations
   - `3` – Image Smoothing
   - `0` – Exit

3. For each category, sub-menu:

   - Color:

     - brightness / contrast / brightness+contrast
     - log / exponential
     - histogram equalization / specification

   - Geometry:

     - scale (custom `sx`, `sy`)
     - rotate (custom angle)
     - translate (custom `tx`, `ty`)
     - shear (custom `kx`, `ky`)

   - Smoothing:

     - mean filter (`ksize`)
     - Gaussian (`ksize`, `sigma`)
     - median (`ksize`)

The user then enters a short output name, and the result is written into the `outputs/interactive` folder.

**Example:**

```bash
python main.py --mode interactive
```

This mode is meant for manual exploration during report writing, and for Agents that need fine-grained control over which algorithm to execute.

---

## 4. Input / Output Contract (for Agents)

- **Input image**: any format that `cv2.imread` supports (e.g. PNG, JPG). Color images are read as BGR.
- **Outputs**:

  - images saved as `.png` inside `--out_dir` with descriptive filenames;
  - in geometric part, additional comparison collages are also generated under `--out_dir/compare/` (scale/rotate/shear/translate).

Agents can either:

- `subprocess.run(["python", "main.py", ...])`, or
- directly import the modules and call:

  - `color_transforms(image)`,
  - `run_geometric_experiments(image, out_dir)`,
  - `smoothing_filters(image)`.

---

## 5. Extending

To add a new algorithm:

1. Implement it in the appropriate module (`color_transform.py`, `geometric_transforms.py`, or `smoothing.py`).
2. Expose it in:

   - the corresponding helper dict in `main.py` (for interactive mode), and
   - optionally in the batch/test pipelines.

Follow the existing naming convention so that it is easy to map functions in the report to functions in the code.

````

---

## 6. Cách chạy

1. **Chuẩn bị môi trường**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate           # Windows
   pip install -r requirements.txt  # hoặc pip install numpy opencv-python
   ```
2. **Chạy nhanh với Lena mặc định**
   ```bash
   python main.py --mode test
   ```
3. **Batch với ảnh tùy chọn**
   ```bash
   python main.py --mode batch --image path\to\your_image.jpg --out_dir outputs_custom
   ```
4. **Interactive mode**
   ```bash
   python main.py --mode interactive --out_dir outputs_playground
   ```
   - Menu CLI cho phép đổi ảnh nguồn, chọn nhóm thuật toán, nhập tham số và đặt tên file.
5. **Ghi chú**
   - Output mặc định nằm trong `outputs/` theo cây con `color/`, `geometric_transforms_demo/`, `smoothing/`, `interactive_*`.
   - Nếu thiếu `assets/Lenna.jpg`, script tự tải về lần đầu tiên chạy.

---

## 7. Cấu trúc Feature

| Feature | File/Hàm chính | Output mặc định |
| --- | --- | --- |
| Color transformations | `color_transform.py`, `color_transforms()` trong `main.py` | `outputs/color/*.png`, `outputs/interactive_color/*.png` |
| Geometric transformations | `geometric_transforms.py`, `run_geometric_experiments`, `warp_affine_*` | `outputs/geometric_transforms_demo/**`, `outputs/interactive_geometry/*.png` |
| Smoothing filters | `smoothing.py`, `smoothing_filters()` | `outputs/smoothing/*.png`, `outputs/interactive_smoothing/*.png` |
| CLI runner | `main.py` (`run_mode_test/batch/interactive`) | Điều phối toàn bộ pipeline + log |
| Assets & data | `assets/Lenna.jpg`, thư mục `outputs/` | Nguồn test chuẩn và kết quả |

> Khi bổ sung feature mới, chỉ cần thêm module tương ứng, expose trong `main.py`, và cập nhật bảng trên nếu cần.

---

## 2. `main.py` refactor – 3 run modes + 2 kiểu input

File `main.py` chứa toàn bộ logic CLI cho ba chế độ `test`, `batch`, `interactive` cùng các helper liên quan (parse arg, tải Lenna, dispatch pipeline). Để xem chi tiết, mở trực tiếp `main.py`; README chỉ mô tả hành vi và cách chạy để tránh trùng lặp mã.

---
````
