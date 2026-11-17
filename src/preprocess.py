"""Image Preprocessing Module (From Scratch with NumPy).

Module thực hiện các phép tiền xử lý ảnh cơ bản từ đầu bằng NumPy:
1. Chuyển đổi không gian màu: RGB -> Grayscale, RGB -> HSV.
2. Tích chập 2 chiều và các bộ lọc làm mịn loại bỏ nhiễu: Mean Filter, Gaussian Filter.

QUY TẮC BẮT BUỘC (CODE FROM SCRATCH):
- TUYỆT ĐỐI KHÔNG sử dụng các hàm xử lý ảnh có sẵn của OpenCV như `cv2.cvtColor`,
  `cv2.GaussianBlur`, `cv2.blur`, `cv2.filter2D`.
- Tất cả các phép toán phải được vector hóa bằng thư viện NumPy.

Ghi chú tối ưu:
- `convolve2d_scratch` dùng kỹ thuật "shift-and-add": lặp qua k*k vị trí của kernel
  (tối đa 121 lần với k = 11), mỗi lần cộng dồn cả một lát cắt ảnh đã đệm biên.
  Mọi phép nhân/cộng trên điểm ảnh đều vector hóa, bộ nhớ phụ chỉ O(H*W).
- Gauss và Mean là kernel TÁCH ĐƯỢC (separable): K = k_col ⊗ k_row, nên ta tích chập
  2 lần 1D (cột rồi hàng) với chi phí O(2k) thay vì O(k²) cho mỗi pixel.
  Kết quả trùng với tích chập 2D trực tiếp (sai khác chỉ ở mức sai số dấu phẩy động).
"""

from typing import Tuple, Union
import numpy as np

# Hệ số độ chói ITU-R BT.601 (cùng chuẩn với OpenCV COLOR_RGB2GRAY)
_BT601_WEIGHTS = (0.299, 0.587, 0.114)

# Ánh xạ tên padding -> mode của np.pad.
# 'reflect' của NumPy = BORDER_REFLECT_101 của OpenCV (gdcba|abcdefgh|gfedcb) — mặc định của cv2.
_PADDING_MODES = {
    "reflect": "reflect",
    "reflect101": "reflect",
    "symmetric": "symmetric",  # = BORDER_REFLECT (dcba|abcd|dcba)
    "edge": "edge",            # = BORDER_REPLICATE
    "replicate": "edge",
    "constant": "constant",    # đệm 0
    "zero": "constant",
}


def _validate_rgb(rgb_img: np.ndarray, func_name: str) -> None:
    if not isinstance(rgb_img, np.ndarray):
        raise ValueError(f"{func_name}: đầu vào phải là np.ndarray, nhận được {type(rgb_img).__name__}.")
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError(f"{func_name}: ảnh phải có shape (H, W, 3), nhận được {rgb_img.shape}.")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Làm tròn, cắt về [0, 255] và ép kiểu uint8 (tránh tràn số khi ép trực tiếp)."""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


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

    Note:
        - QUY TẮC: Không dùng `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)`.
        - Ép kiểu trung gian sang `float64` trước khi nhân hệ số để tránh tràn số
          trên mảng `uint8`, sau đó làm tròn và ép lại `np.uint8`.
        - Sai khác so với OpenCV tối đa ±1 mức xám (OpenCV dùng số học dấu phẩy tĩnh 14-bit).
    """
    _validate_rgb(rgb_img, "rgb_to_gray_scratch")
    rgb = rgb_img.astype(np.float64)
    wr, wg, wb = _BT601_WEIGHTS
    gray = wr * rgb[:, :, 0] + wg * rgb[:, :, 1] + wb * rgb[:, :, 2]
    return _to_uint8(gray)


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

    Note:
        - QUY TẮC: Không dùng `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)`.
        - Thang H [0..255] tương đương `cv2.COLOR_RGB2HSV_FULL` (không phải [0..180] mặc định).
        - Hue là đại lượng VÒNG: H = 0 và H = 255 đều là màu đỏ (gần 0°/360°).
        - Trường hợp chia cho 0 (C = 0 hoặc V = 0) được xử lý bằng mặt nạ boolean.
    """
    _validate_rgb(rgb_img, "rgb_to_hsv_scratch")
    rgb = rgb_img.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    delta = cmax - cmin

    # Mẫu số an toàn: chỗ delta = 0 thay bằng 1 để không sinh NaN/inf (kết quả bị che bởi mask sau đó)
    safe_delta = np.where(delta > 0, delta, 1.0)
    has_chroma = delta > 0

    # Thứ tự ưu tiên khi nhiều kênh cùng đạt max: R -> G -> B (giống OpenCV)
    is_r = has_chroma & (cmax == r)
    is_g = has_chroma & (cmax == g) & ~is_r
    is_b = has_chroma & ~is_r & ~is_g

    hue = np.zeros_like(cmax)
    hue = np.where(is_r, 60.0 * np.mod((g - b) / safe_delta, 6.0), hue)
    hue = np.where(is_g, 60.0 * ((b - r) / safe_delta + 2.0), hue)
    hue = np.where(is_b, 60.0 * ((r - g) / safe_delta + 4.0), hue)

    sat = np.where(cmax > 0, delta / np.where(cmax > 0, cmax, 1.0), 0.0)
    val = cmax

    h_u8 = _to_uint8(hue / 360.0 * 255.0)
    s_u8 = _to_uint8(sat * 255.0)
    v_u8 = _to_uint8(val * 255.0)
    return np.stack([h_u8, s_u8, v_u8], axis=-1)


def create_gaussian_kernel_2d(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Tạo ma trận mặt nạ lọc Gauss (Gaussian Kernel) 2D từ đầu bằng công thức phân phối chuẩn.

    Công thức hàm Gauss 2D đối xứng qua tâm:
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(- (x^2 + y^2) / (2 * sigma^2))
    Kernel sau đó được chuẩn hóa sao cho tổng toàn bộ các phần tử bằng 1
    (vì vậy hằng số 1 / (2*pi*sigma^2) bị triệt tiêu và không cần tính).

    Args:
        kernel_size: Kích thước cạnh ma trận lọc vuông (phải là số nguyên lẻ >= 3, ví dụ: 3, 5, 7).
        sigma: Độ lệch chuẩn (standard deviation) của phân phối Gauss (sigma > 0).

    Returns:
        np.ndarray: Ma trận trọng số 2 chiều kích thước `(kernel_size, kernel_size)`,
            kiểu dữ liệu `np.float64`, tổng các phần tử bằng 1.0.

    Raises:
        ValueError: Nếu `kernel_size` là số chẵn hoặc nhỏ hơn 3, hoặc `sigma <= 0`.
    """
    k1d = _gaussian_kernel_1d(kernel_size, sigma)
    # Gauss 2D đẳng hướng = tích ngoài của 2 vector Gauss 1D (tính chất tách được)
    kernel = np.outer(k1d, k1d)
    return kernel / np.sum(kernel)


def _validate_kernel_size(kernel_size: int) -> int:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError(f"kernel_size phải là số nguyên, nhận được {kernel_size!r}.")
    kernel_size = int(kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size phải là số nguyên lẻ >= 3, nhận được {kernel_size}.")
    return kernel_size


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Vector Gauss 1D đã chuẩn hóa tổng = 1: g(x) ∝ exp(-x² / (2σ²))."""
    kernel_size = _validate_kernel_size(kernel_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận được {sigma}.")
    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    return g / np.sum(g)


def _resolve_padding(padding_mode: str) -> str:
    key = str(padding_mode).lower()
    if key not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode '{padding_mode}' không hỗ trợ. Chọn một trong: {sorted(_PADDING_MODES)}."
        )
    return _PADDING_MODES[key]


def _pad_spatial(image: np.ndarray, pad_h: int, pad_w: int, np_mode: str) -> np.ndarray:
    """Đệm biên chỉ trên 2 trục không gian (H, W), giữ nguyên trục kênh màu nếu có."""
    pad_width = [(pad_h, pad_h), (pad_w, pad_w)] + [(0, 0)] * (image.ndim - 2)
    if np_mode == "constant":
        return np.pad(image, pad_width, mode="constant", constant_values=0)
    return np.pad(image, pad_width, mode=np_mode)


def _correlate_shift_add(padded: np.ndarray, kernel: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Tương quan (correlation) bằng shift-and-add: out = Σ_{i,j} K[i, j] * padded[i:i+H, j:j+W]."""
    kh, kw = kernel.shape
    out = np.zeros((out_h, out_w) + padded.shape[2:], dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w != 0.0:
                out += w * padded[i:i + out_h, j:j + out_w]
    return out


def convolve2d_scratch(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """Thực hiện phép tích chập không gian 2 chiều (2D Spatial Convolution) từ đầu bằng NumPy.

    Hỗ trợ xử lý cho cả ảnh đơn kênh mức xám `(H, W)` lẫn ảnh màu đa kênh `(H, W, C)`
    (kernel được áp dụng độc lập trên từng kênh, vector hóa đồng thời mọi kênh).

    Định nghĩa tích chập (convolution) đúng toán học:
        out(x, y) = Σ_i Σ_j K(i, j) * I(x - i, y - j)
    tương đương tương quan (correlation) với kernel đã lật 180°. Với kernel đối xứng
    (Gauss, Mean) hai phép này cho kết quả như nhau.

    Args:
        image: Mảng ảnh đầu vào, kích thước `(H, W)` hoặc `(H, W, C)`, kiểu `np.uint8` hoặc `np.float32`.
        kernel: Ma trận bộ lọc 2D kích thước `(kh, kw)` với kh, kw lẻ, kiểu `np.float64`.
        padding_mode: Phương thức đệm biên ảnh ('reflect', 'constant', 'edge', 'symmetric').
            Mặc định 'reflect' (= BORDER_REFLECT_101, mặc định của OpenCV).

    Returns:
        np.ndarray: Mảng ảnh kết quả sau tích chập, cùng kích thước `shape` với ảnh gốc,
            kiểu dữ liệu `np.uint8`, giá trị được clip trong đoạn `[0, 255]`.

    Raises:
        ValueError: Nếu kernel không phải 2D, có cạnh chẵn hoặc ảnh không phải 2D/3D.
            (Ảnh nhỏ hơn kernel vẫn hợp lệ: np.pad phản xạ lặp lại nhiều lần.)

    Note:
        - QUY TẮC: Không dùng `scipy.signal.convolve2d` hay `cv2.filter2D`.
        - Áp dụng same padding để ảnh đầu ra có kích thước nguyên vẹn.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError(f"image phải là mảng (H, W) hoặc (H, W, C), nhận được {getattr(image, 'shape', None)}.")
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"kernel phải là ma trận 2D, nhận được shape {kernel.shape}.")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"Kích thước kernel phải lẻ ở cả 2 chiều, nhận được {kernel.shape}.")

    np_mode = _resolve_padding(padding_mode)
    pad_h, pad_w = kh // 2, kw // 2
    H, W = image.shape[:2]

    padded = _pad_spatial(image.astype(np.float64), pad_h, pad_w, np_mode)
    flipped = kernel[::-1, ::-1]  # lật kernel => tích chập đúng nghĩa
    out = _correlate_shift_add(padded, flipped, H, W)
    return _to_uint8(out)


def _separable_filter(
    img: np.ndarray,
    k1d: np.ndarray,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """Lọc tách được: tích chập 1D theo trục dọc rồi trục ngang (kernel đối xứng)."""
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError(f"Ảnh phải có shape (H, W) hoặc (H, W, C), nhận được {getattr(img, 'shape', None)}.")
    np_mode = _resolve_padding(padding_mode)
    k = k1d.size
    half = k // 2
    H, W = img.shape[:2]

    padded = _pad_spatial(img.astype(np.float64), half, half, np_mode)
    # Pass 1 (dọc): giữ nguyên phần đệm ngang để pass 2 dùng
    tmp = np.zeros((H, padded.shape[1]) + padded.shape[2:], dtype=np.float64)
    for i in range(k):
        tmp += k1d[i] * padded[i:i + H]
    # Pass 2 (ngang)
    out = np.zeros((H, W) + padded.shape[2:], dtype=np.float64)
    for j in range(k):
        out += k1d[j] * tmp[:, j:j + W]
    return _to_uint8(out)


def gaussian_filter_scratch(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Bộ lọc làm mịn Gauss (Gaussian Blur) từ đầu bằng NumPy để giảm nhiễu hạt trước khi phân đoạn.

    Cài đặt theo dạng tách được (2 lần tích chập 1D), cho kết quả tương đương tích chập
    với `create_gaussian_kernel_2d(kernel_size, sigma)` nhưng nhanh hơn ~k/2 lần.

    Args:
        img: Mảng ảnh 2D `(H, W)` hoặc 3D `(H, W, C)`, kiểu `np.uint8`.
        kernel_size: Kích thước mặt nạ lọc lẻ (ví dụ: 3, 5, 7). Mặc định là 5.
            Giá trị 1 được hiểu là "không lọc" và trả về bản sao ảnh.
        sigma: Độ lệch chuẩn của hàm Gauss. Mặc định là 1.0.

    Returns:
        np.ndarray: Ảnh đã được làm mịn, cùng kích thước và kiểu `np.uint8` với ảnh đầu vào.

    Raises:
        ValueError: Nếu kernel_size chẵn/âm, sigma <= 0 hoặc ảnh sai shape.
    """
    if kernel_size == 1:
        return _to_uint8(np.asarray(img, dtype=np.float64))
    k1d = _gaussian_kernel_1d(kernel_size, sigma)
    return _separable_filter(img, k1d, padding_mode="reflect")


def mean_filter_scratch(
    img: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Bộ lọc trung bình (Mean / Box Filter) từ đầu bằng NumPy để khử nhiễu đều.

    Ma trận lọc đồng nhất kích thước `(kernel_size, kernel_size)` với mỗi trọng số bằng
    `1 / (kernel_size^2)` = tích ngoài của 2 vector `1/k`, nên cũng được tính tách được.

    Args:
        img: Mảng ảnh 2D `(H, W)` hoặc 3D `(H, W, C)`, kiểu `np.uint8`.
        kernel_size: Kích thước mặt nạ lọc lẻ (ví dụ: 3, 5). Mặc định là 3.

    Returns:
        np.ndarray: Ảnh làm mịn, cùng kích thước và kiểu `np.uint8` với ảnh đầu vào.

    Raises:
        ValueError: Nếu kernel_size chẵn/nhỏ hơn 3 hoặc ảnh sai shape.
    """
    k = _validate_kernel_size(kernel_size)
    k1d = np.full(k, 1.0 / k, dtype=np.float64)
    return _separable_filter(img, k1d, padding_mode="reflect")
