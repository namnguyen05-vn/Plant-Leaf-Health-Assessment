"""Image Preprocessing Module (From Scratch with NumPy).

Module thực hiện các phép tiền xử lý ảnh cơ bản từ đầu bằng NumPy:
1. Chuyển đổi không gian màu: RGB -> Grayscale, RGB -> HSV.
2. Tích chập 2 chiều và các bộ lọc làm mịn loại bỏ nhiễu: Mean Filter, Gaussian Filter.

QUY TẮC BẮT BUỘC (CODE FROM SCRATCH):
- TUYỆT ĐỐI KHÔNG sử dụng các hàm xử lý ảnh có sẵn của OpenCV như `cv2.cvtColor`,
  `cv2.GaussianBlur`, `cv2.blur`, `cv2.filter2D`.
- Tất cả các phép toán phải được vector hóa bằng thư viện NumPy.
"""

from typing import Tuple, Union
import numpy as np


def rgb_to_gray_scratch(rgb_img: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh màu RGB sang ảnh mức xám Grayscale từ đầu bằng công thức độ chói chuẩn ITU-R BT.601.

    Công thức chuyển đổi chuẩn:
        Y = round(0.299 * R + 0.587 * G + 0.114 * B)

    Args:
        rgb_img: Mảng 3 chiều không gian màu RGB, kích thước `(H, W, 3)`,
            kiểu dữ liệu `np.uint8`, giá trị trong đoạn `[0, 255]`.

    Returns:
        np.ndarray: Mảng 2 chiều biểu diễn ảnh mức xám, kích thước `(H, W)`,
            kiểu dữ liệu `np.uint8`, giá trị trong đoạn `[0, 255]`.

    Raises:
        ValueError: Nếu ảnh đầu vào không có đúng 3 chiều hoặc kênh màu khác 3.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Không dùng `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)`.
        - Cần thực hiện phép ép kiểu trung gian sang `float32` hoặc `float64` trước khi nhân hệ số
          để tránh hiện tượng tràn số (overflow) trên mảng `uint8`, sau đó làm tròn và ép lại `np.uint8`.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Kiểm tra `rgb_img.ndim == 3` và `rgb_img.shape[2] == 3`.
    # Bước 2: Tách các kênh: R = rgb_img[:, :, 0], G = rgb_img[:, :, 1], B = rgb_img[:, :, 2].
    # Bước 3: Áp dụng công thức vector hóa:
    #         gray = 0.299 * R + 0.587 * G + 0.114 * B
    # Bước 4: Làm tròn (np.round) và clip giá trị trong khoảng [0, 255], chuyển sang np.uint8.
    # =========================================================================
    raise NotImplementedError("Hàm rgb_to_gray_scratch cần được cài đặt trong nhánh feat/preprocess.")


def rgb_to_hsv_scratch(rgb_img: np.ndarray) -> np.ndarray:
    """Chuyển đổi không gian màu từ RGB sang HSV từ đầu bằng công thức toán học và NumPy vectorization.

    Các thành phần chuyển đổi:
    - R, G, B được chuẩn hóa về miền [0, 1].
    - V = max(R, G, B)
    - C = V - min(R, G, B) (Chroma)
    - S = C / V nếu V > 0, ngược lại S = 0
    - H (Hue góc 0°..360°) tính dựa trên kênh đạt cực đại:
        * Nếu C == 0: H = 0
        * Nếu V == R: H = 60° * ((G - B) / C mod 6)
        * Nếu V == G: H = 60° * ((B - R) / C + 2)
        * Nếu V == B: H = 60° * ((R - G) / C + 4)
    - Chuẩn hóa đầu ra về thang `[0, 255]`:
        * H_norm = round((H / 360.0) * 255.0)
        * S_norm = round(S * 255.0)
        * V_norm = round(V * 255.0)

    Args:
        rgb_img: Mảng 3 chiều RGB kích thước `(H, W, 3)`, kiểu `np.uint8`, dải `[0, 255]`.

    Returns:
        np.ndarray: Mảng 3 chiều HSV kích thước `(H, W, 3)`, kiểu `np.uint8`,
            trong đó cả 3 kênh H, S, V đều được chuẩn hóa nằm trong khoảng `[0, 255]`.

    Raises:
        ValueError: Nếu kích thước không đúng định dạng `(H, W, 3)`.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Không dùng `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)`.
        - Cần xử lý cẩn thận các trường hợp chia cho 0 khi Chroma C = 0 hoặc V = 0 bằng `np.where`.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Chuẩn hóa rgb_img sang float trong [0, 1].
    # Bước 2: Tìm Cmax, Cmin và Delta = Cmax - Cmin.
    # Bước 3: Tính toán vector hóa H, S, V theo công thức toán học.
    # Bước 4: Scale H về [0..255], S về [0..255], V về [0..255].
    # Bước 5: Ghép các kênh lại thành mảng (H, W, 3) kiểu np.uint8.
    # =========================================================================
    raise NotImplementedError("Hàm rgb_to_hsv_scratch cần được cài đặt trong nhánh feat/preprocess.")


def create_gaussian_kernel_2d(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Tạo ma trận mặt nạ lọc Gauss (Gaussian Kernel) 2D từ đầu bằng công thức phân phối chuẩn.

    Công thức hàm Gauss 2D đối xứng qua tâm:
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(- (x^2 + y^2) / (2 * sigma^2))
    Kernel sau đó được chuẩn hóa sao cho tổng toàn bộ các phần tử bằng 1.

    Args:
        kernel_size: Kích thước cạnh ma trận lọc vuông (phải là số nguyên lẻ >= 3, ví dụ: 3, 5, 7).
        sigma: Độ lệch chuẩn (standard deviation) của phân phối Gauss (sigma > 0).

    Returns:
        np.ndarray: Ma trận trọng số 2 chiều kích thước `(kernel_size, kernel_size)`,
            kiểu dữ liệu `np.float64`, tổng các phần tử bằng 1.0.

    Raises:
        ValueError: Nếu `kernel_size` là số chẵn hoặc nhỏ hơn 3, hoặc `sigma <= 0`.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Kiểm tra kernel_size % 2 == 1 và sigma > 0.
    # Bước 2: Tạo lưới tọa độ x, y cách tâm: k = kernel_size // 2; ax = np.arange(-k, k + 1).
    # Bước 3: Tính ma trận khoảng cách: xx, yy = np.meshgrid(ax, ax); g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).
    # Bước 4: Chuẩn hóa kernel: kernel = g / np.sum(g). Trả về kernel dạng float64.
    # =========================================================================
    raise NotImplementedError("Hàm create_gaussian_kernel_2d cần được cài đặt trong nhánh feat/preprocess.")


def convolve2d_scratch(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """Thực hiện phép tích chập không gian 2 chiều (2D Spatial Convolution) từ đầu bằng NumPy.

    Hỗ trợ xử lý cho cả ảnh đơn kênh mức xám `(H, W)` lẫn ảnh màu đa kênh `(H, W, C)`.

    Args:
        image: Mảng ảnh đầu vào, kích thước `(H, W)` hoặc `(H, W, C)`, kiểu `np.uint8` hoặc `np.float32`.
        kernel: Ma trận bộ lọc vuông 2D kích thước `(k, k)`, kiểu `np.float64`.
        padding_mode: Phương thức đệm biên ảnh ('reflect', 'constant', 'edge'). Mặc định 'reflect'.

    Returns:
        np.ndarray: Mảng ảnh kết quả sau tích chập, cùng kích thước `shape` với ảnh gốc,
            kiểu dữ liệu `np.uint8`, giá trị được clip trong đoạn `[0, 255]`.

    Raises:
        ValueError: Nếu kích thước kernel không đối xứng hoặc không phải 2D.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Không dùng `scipy.signal.convolve2d` hay `cv2.filter2D`.
        - Cần áp dụng kỹ thuật padding để ảnh đầu ra có kích thước nguyên vẹn (same padding).
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Tính bán kính đệm pad_w = kernel.shape[0] // 2.
    # Bước 2: Đệm biên ảnh theo padding_mode (dùng np.pad).
    # Bước 3: Nếu là ảnh 3D (H, W, C), lặp qua từng kênh hoặc vectorized sliding window.
    # Bước 4: Nhân chập (element-wise multiplication) và tính tổng cho từng cửa sổ trượt.
    # Bước 5: Clip giá trị về [0, 255] và ép kiểu về np.uint8.
    # =========================================================================
    raise NotImplementedError("Hàm convolve2d_scratch cần được cài đặt trong nhánh feat/preprocess.")


def gaussian_filter_scratch(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Bộ lọc làm mịn Gauss (Gaussian Blur) từ đầu bằng NumPy để giảm nhiễu hạt trước khi phân đoạn.

    Args:
        img: Mảng ảnh 2D `(H, W)` hoặc 3D `(H, W, C)`, kiểu `np.uint8`.
        kernel_size: Kích thước mặt nạ lọc lẻ (ví dụ: 3, 5, 7). Mặc định là 5.
        sigma: Độ lệch chuẩn của hàm Gauss. Mặc định là 1.0.

    Returns:
        np.ndarray: Ảnh đã được làm mịn, cùng kích thước và kiểu `np.uint8` với ảnh đầu vào.

    Raises:
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Tạo kernel Gauss bằng hàm `create_gaussian_kernel_2d(kernel_size, sigma)`.
    # Bước 2: Gọi hàm `convolve2d_scratch(img, kernel)` để thu được ảnh làm mịn.
    # =========================================================================
    raise NotImplementedError("Hàm gaussian_filter_scratch cần được cài đặt trong nhánh feat/preprocess.")


def mean_filter_scratch(
    img: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Bộ lọc trung bình (Mean / Box Filter) từ đầu bằng NumPy để khử nhiễu đều.

    Tạo ma trận lọc đồng nhất kích thước `(kernel_size, kernel_size)` với mỗi trọng số bằng `1 / (kernel_size^2)`.

    Args:
        img: Mảng ảnh 2D `(H, W)` hoặc 3D `(H, W, C)`, kiểu `np.uint8`.
        kernel_size: Kích thước mặt nạ lọc lẻ (ví dụ: 3, 5). Mặc định là 3.

    Returns:
        np.ndarray: Ảnh làm mịn, cùng kích thước và kiểu `np.uint8` với ảnh đầu vào.

    Raises:
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Tạo ma trận kernel đồng nhất: np.ones((k, k), dtype=np.float64) / (k * k).
    # Bước 2: Gọi `convolve2d_scratch(img, kernel)`.
    # =========================================================================
    raise NotImplementedError("Hàm mean_filter_scratch cần được cài đặt trong nhánh feat/preprocess.")
