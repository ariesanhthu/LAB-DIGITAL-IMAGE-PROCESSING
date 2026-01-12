# Edge Detection Project

Project phát hiện biên ảnh sử dụng các thuật toán cổ điển (classical) và deep learning.

## 📋 Mục lục

- [Cấu trúc Project](#cấu-trúc-project)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Setup Dataset](#setup-dataset)
- [Sử dụng](#sử-dụng)
  - [Classical Edge Detection](#classical-edge-detection)
  - [Evaluation Metrics](#evaluation-metrics)
- [Cấu trúc Code](#cấu-trúc-code)

## 📁 Cấu trúc Project

```
source/
├── classical/              # Thuật toán cổ điển
│   ├── base.py            # Base class cho edge detectors
│   ├── gradient.py        # Gradient operators (Basic, Differencing, Roberts, Prewitt, Sobel, Frei-Chen)
│   ├── laplacian.py       # Laplacian operators
│   ├── log.py             # Laplacian of Gaussian
│   └── canny.py           # Canny edge detector
├── deep_learning/         # Deep learning models (inference only)
│   ├── test_hed.py        # HED model loading và inference với OpenCV DNN
│   └── __init__.py        # Module exports
├── evaluation/            # Evaluation scripts
│   ├── test_classical.py  # Test và evaluation cho classical algorithms trên BIPED dataset
│   ├── evaluate_deep_models.py  # Evaluate HED và U-Net models trên BIPED dataset
│   └── evaluation.py      # Evaluation metrics table và Precision-Recall curves
├── utils/                 # Utilities
│   ├── image_utils.py     # Image I/O và preprocessing
│   └── visualization.py   # Visualization functions
└── main.py                # Entry point - chạy traditional edge detection trên một ảnh
```

## 💻 Yêu cầu hệ thống

- Python 3.7+
- CUDA (tùy chọn, để train trên GPU)
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- Disk: ~5GB cho dataset và checkpoints

## 🔧 Cài đặt

### 1. Cài đặt Python dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt PyTorch (chọn version phù hợp với hệ thống)
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các dependencies khác
pip install numpy pillow opencv-python tqdm matplotlib scikit-image scipy
```

### 2. Kiểm tra cài đặt

```bash
# Chạy từ thư mục source/
python main.py
```

Nếu không có lỗi và các file kết quả được tạo tại `results/classical/`, cài đặt thành công!

**Lưu ý:** Đảm bảo có file ảnh tại `source/data/RGB_008.jpg` hoặc dùng `--image` để chỉ định đường dẫn ảnh khác.

## 📦 Setup Dataset

### BIPED Dataset

Project sử dụng BIPED dataset. Cấu trúc dataset:

```
dataset/
└── BIPED/
    └── edges/
        ├── imgs/
        │   ├── train/
        │   │   └── rgbr/
        │   │       └── real/        # Training images
        │   └── test/
        │       └── rgbr/            # Test images
        ├── edge_maps/
        │   ├── train/
        │   │   └── rgbr/
        │   │       └── real/         # Training labels
        │   └── test/
        │       └── rgbr/             # Test labels
        ├── train_rgb.lst             # Training file list
        └── test_rgb.lst               # Test file list
```

**Lưu ý:** Đảm bảo dataset đã được download và đặt đúng cấu trúc như trên.

## 🚀 Sử dụng

### Chạy traditional edge detection một ảnh (main.py)

- Mặc định sẽ chạy **TẤT CẢ** traditional edge detection algorithms, đọc ảnh `source/data/RGB_008.jpg`, lưu kết quả vào `results/classical/`.

```bash
cd source
python main.py                             # chạy TẤT CẢ detectors với ảnh mặc định
python main.py --image path/to/img.jpg     # ảnh tùy chọn
python main.py --detector sobel            # chỉ chạy Sobel
python main.py --detector canny            # chỉ chạy Canny
python main.py --detector roberts          # chỉ chạy Roberts
python main.py --detector log --sigma 2.0  # chỉ chạy LoG với sigma=2.0
python main.py --detector canny --sigma 1.5 --low_threshold 0.1 --high_threshold 0.3
python main.py --output_dir results/my_classical  # thay đổi thư mục output
```

**Các detectors có sẵn (12 detectors):**

**Basic & Differencing Operators:**

- `basic_gradient`: Basic gradient operator (fx, fy, magnitude, direction)
- `forward_diff`: Forward difference operator
- `backward_diff`: Backward difference operator
- `central_diff`: Central difference operator

**Gradient-based Operators:**

- `roberts`: Roberts cross operator
- `prewitt`: Prewitt operator
- `sobel`: Sobel operator
- `freichen`: Frei-Chen operator

**Laplacian-based Operators:**

- `laplacian4`: Laplacian 4-neighborhood
- `laplacian8`: Laplacian 8-neighborhood
- `log`: Laplacian of Gaussian (cần `--sigma`)

**Advanced:**

- `canny`: Canny edge detector (cần `--sigma`, `--low_threshold`, `--high_threshold`)

**Lưu ý:** Khi không chỉ định `--detector`, script sẽ chạy tất cả 12 detectors và lưu kết quả vào các file riêng biệt trong `results/classical/`.

## 📊 Evaluation Scripts

Project cung cấp các script evaluation trong thư mục `evaluation/` để đánh giá và so sánh các phương pháp edge detection trên BIPED dataset.

### 1. Test Classical Algorithms (`test_classical.py`)

Test và evaluation cho các classical edge detection algorithms trên BIPED dataset.

#### Cách chạy:

```bash
# Từ thư mục source/
cd source

# Test tất cả algorithms trên toàn bộ test set
python evaluation/test_classical.py

# Test với số lượng samples giới hạn (nhanh hơn)
python evaluation/test_classical.py --max_samples 10

# Test và lưu ảnh kết quả
python evaluation/test_classical.py --save_images

# Custom threshold và output directory
python evaluation/test_classical.py --threshold 100 --output_dir results/my_test

# Custom dataset path
python evaluation/test_classical.py --dataset_root ../dataset/BIPED/edges
```

#### Các tham số:

- `--dataset_root`: Root directory của BIPED dataset (mặc định: `dataset/BIPED/edges`)
- `--output_dir`: Thư mục lưu kết quả (mặc định: `results/classical`)
- `--threshold`: Threshold để binarize edge maps (0-255, mặc định: 128)
- `--max_samples`: Số lượng samples tối đa để test (None = tất cả)
- `--save_images`: Lưu ảnh kết quả cho một số samples
- `--no_plot`: Không tạo biểu đồ so sánh

#### Kết quả:

- `results.json`: Metrics (Precision, Recall, F1, IoU) cho từng algorithm
- `metrics_comparison.png`: Biểu đồ so sánh metrics
- `images/sample_*/`: Ảnh kết quả cho các samples (nếu `--save_images`)

### 2. Evaluate Deep Learning Models (`evaluate_deep_models.py`)

Evaluate các deep learning models (HED và U-Net) trên BIPED dataset và tạo bảng metrics cùng biểu đồ so sánh.

#### Yêu cầu:

- U-Net checkpoint: `source/model/biped_edge_unet_best.pth`
- HED model files:
  - `UNet_edge_detection/deploy.prototxt.txt` hoặc `source/model/deploy.prototxt.txt`
  - `UNet_edge_detection/hed_pretrained_bsds.caffemodel` hoặc `source/model/hed_pretrained_bsds.caffemodel`

#### Cách chạy:

```bash
# Từ thư mục source/
cd source

# Chạy evaluation (mặc định test trên 5 ảnh)
python evaluation/evaluate_deep_models.py
```

#### Kết quả:

Kết quả được lưu tại `source/results/deep_learning/`:

- `deep_models_metrics.csv`: Bảng metrics dạng CSV
- `deep_models_metrics_comparison.png`: Biểu đồ so sánh F1, Precision, Recall, IoU
- `deep_models_time_comparison.png`: Biểu đồ so sánh thời gian inference

**Metrics được tính:** F1 Score, Precision, Recall, IoU, Time (ms)

### 3. Evaluation Metrics Table và PR Curves (`evaluation.py`)

Script cung cấp các hàm để đánh giá và so sánh các phương pháp edge detection với metrics table và Precision-Recall curves.

#### Cách chạy:

**Cách 1: Chạy trực tiếp từ terminal**

```bash
# Từ thư mục root project (LAB2)
python -m source.evaluation.evaluation

# Hoặc từ thư mục source/
cd source
python -m evaluation.evaluation
```

**Cách 2: Import trong Python script/notebook**

```python
# Từ root project
from source.evaluation.evaluation import test_metrics_table, test_biped_evaluation

# Chạy metrics table evaluation trên 5 ảnh
test_metrics_table()

# Hoặc chạy PR curves evaluation trên 10 ảnh
test_biped_evaluation()
```

**Cách 3: Sử dụng các hàm riêng lẻ**

```python
from source.evaluation.evaluation import evaluate_metrics_table, print_metrics_table

# Tính metrics trên số lượng ảnh tùy chọn
metrics_table = evaluate_metrics_table(
    biped_root="dataset/BIPED/edges",
    max_images=5,          # Số lượng ảnh để test (None = tất cả)
    threshold=127.5        # Ngưỡng để binarize prediction (0-255)
)

# In bảng kết quả
print_metrics_table(metrics_table)
```

#### Kết quả Metrics Table:

Bảng metrics hiển thị các thông tin sau cho mỗi phương pháp:

| Method        | F1  | Precision | Recall | IoU | Time (ms) |
| ------------- | --- | --------- | ------ | --- | --------- |
| BasicGradient | ... | ...       | ...    | ... | ...       |
| ForwardDiff   | ... | ...       | ...    | ... | ...       |
| BackwardDiff  | ... | ...       | ...    | ... | ...       |
| CentralDiff   | ... | ...       | ...    | ... | ...       |
| Roberts       | ... | ...       | ...    | ... | ...       |
| Prewitt       | ... | ...       | ...    | ... | ...       |
| Sobel         | ... | ...       | ...    | ... | ...       |
| FreiChen      | ... | ...       | ...    | ... | ...       |
| Laplacian4    | ... | ...       | ...    | ... | ...       |
| Laplacian8    | ... | ...       | ...    | ... | ...       |
| LapVar1-4     | ... | ...       | ...    | ... | ...       |
| Canny         | ... | ...       | ...    | ... | ...       |

#### Các hàm có sẵn:

- **`test_metrics_table()`**: Hàm tiện ích để chạy nhanh evaluation trên 5 ảnh và in bảng metrics
- **`test_biped_evaluation()`**: Tính và vẽ Precision-Recall curves (sử dụng `evaluate_classical_and_deep_on_biped()`)
- **`evaluate_metrics_table()`**: Tính metrics cho tất cả các phương pháp classical
- **`print_metrics_table()`**: In bảng kết quả dạng text table
- **`evaluate_classical_and_deep_on_biped()`**: Evaluate cả classical và deep learning models, trả về PR curves
- **`plot_pr_curves()`**: Vẽ Precision-Recall curves cho nhiều phương pháp

#### Ví dụ sử dụng nâng cao:

```python
from source.evaluation.evaluation import evaluate_metrics_table, print_metrics_table

# Test trên 10 ảnh với threshold khác
metrics_table = evaluate_metrics_table(
    biped_root="dataset/BIPED/edges",
    max_images=10,
    threshold=100.0  # Threshold thấp hơn
)

# In và lưu kết quả
print_metrics_table(metrics_table)

# Truy cập metrics của một phương pháp cụ thể
sobel_metrics = metrics_table["Sobel"]
print(f"Sobel F1: {sobel_metrics['f1']:.4f}")
print(f"Sobel Time: {sobel_metrics['time_ms']:.2f} ms")
```

**Lưu ý:**

- Dataset phải được đặt đúng cấu trúc tại `dataset/BIPED/edges/`
- Mặc định `test_metrics_table()` test trên 5 ảnh để chạy nhanh
- Để test trên toàn bộ dataset, đặt `max_images=None` hoặc không truyền tham số này

#### Sử dụng trong code

```python
from classical import RobertsOperator, SobelOperator, CannyEdgeDetector
from utils import load_image, visualize_edge_detection

# Load ảnh
image = load_image("path/to/image.jpg")

# Roberts Operator
roberts = RobertsOperator()
edge_map = roberts(image)
visualize_edge_detection(image, edge_map, "Roberts")

# Sobel Operator
sobel = SobelOperator()
edge_map = sobel(image)
visualize_edge_detection(image, edge_map, "Sobel")

# Canny Edge Detector
canny = CannyEdgeDetector(sigma=1.0, low_threshold=0.1, high_threshold=0.2)
edge_map = canny(image)
visualize_edge_detection(image, edge_map, "Canny")
```

## 📚 Cấu trúc Code

### Classical Algorithms

- **`classical/base.py`**: Base class cho tất cả classical edge detectors
- **`classical/gradient.py`**: Gradient-based operators
  - `BasicGradient`: Basic gradient operator (fx, fy, magnitude, direction)
  - `ForwardDifferenceOperator`: Forward difference operator
  - `BackwardDifferenceOperator`: Backward difference operator
  - `CentralDifferenceOperator`: Central difference operator
  - `RobertsOperator`: Roberts cross operator
  - `PrewittOperator`: Prewitt operator
  - `SobelOperator`: Sobel operator
  - `FreiChenOperator`: Frei-Chen operator
- **`classical/laplacian.py`**: Laplacian operators
  - `Laplacian4Neighbor`: 4-neighborhood Laplacian
  - `Laplacian8Neighbor`: 8-neighborhood Laplacian
- **`classical/log.py`**: `LaplacianOfGaussian` - LoG filter
- **`classical/canny.py`**: `CannyEdgeDetector` - Canny edge detector

### Deep Learning Models

- **`deep_learning/test_hed.py`**: HED model loading và inference với OpenCV DNN
  - `load_hed_caffe()`: Load HED Caffe model
  - `predict_hed_opencv()`: Predict edges với HED model

**Lưu ý:** Module `deep_learning` chỉ chứa code inference. Training được thực hiện trên notebook riêng.

### Utilities

- **`utils/image_utils.py`**: Image I/O, preprocessing, postprocessing
- **`utils/visualization.py`**: Visualization functions

## 📝 Notes

- Dataset BIPED cần được download và đặt đúng cấu trúc như mô tả ở trên.

- Khi chạy `main.py` không có tham số `--detector`, tất cả 12 detectors sẽ được chạy và lưu kết quả vào `results/classical/` với tên file tương ứng (ví dụ: `sobel.png`, `canny.png`, `basic_gradient.png`, ...).

## 🐛 Troubleshooting

Kiểm tra đường dẫn dataset:

```bash
python test_classical.py --dataset_root /path/to/dataset/BIPED/edges
```

### Import errors

Đảm bảo đang chạy từ thư mục `source/` hoặc thêm vào PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/source"
```

### Lỗi không tìm thấy test images

Kiểm tra đường dẫn dataset:

```bash
python evaluation/test_classical.py --dataset_root /path/to/dataset/BIPED/edges
```

### Lỗi không tìm thấy U-Net checkpoint

Đảm bảo file checkpoint tồn tại tại `source/model/biped_edge_unet_best.pth`:

```bash
# Kiểm tra file có tồn tại không
ls source/model/biped_edge_unet_best.pth
```

### Lỗi không tìm thấy HED model files

Đảm bảo các file HED model tồn tại:

- `UNet_edge_detection/deploy.prototxt.txt` hoặc `source/model/deploy.prototxt.txt`
- `UNet_edge_detection/hed_pretrained_bsds.caffemodel` hoặc `source/model/hed_pretrained_bsds.caffemodel`

### Test chạy quá chậm

Giảm số lượng samples để test nhanh hơn:

```bash
python evaluation/test_classical.py --max_samples 10
```

## 📄 License

Project này được tạo cho mục đích học tập và nghiên cứu.
