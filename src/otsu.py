"""Otsu Automatic Thresholding Module (From Scratch with NumPy).

Module triển khai thuật toán phân ngưỡng tự động Otsu 256 mức xám từ đầu:
1. Thống kê biểu đồ tần suất mức xám (Histogram) trên toàn ảnh hoặc trong vùng ROI giới hạn.
2. Tối ưu hóa hàm mục tiêu phương sai giữa các lớp (Between-class variance) để tìm ngưỡng cắt tối ưu T*.
3. Tạo mặt nạ nhị phân tách biệt tiền cảnh (lá cây) và hậu cảnh (background).

Dùng cho cả 2 bài toán của hệ thống:
- Tầng 1 — tách nền lá: `otsu_threshold_scratch(veg_map)` (lá sáng trên bản đồ thực vật).
- Phân ngưỡng đốm bệnh: `otsu_threshold_scratch(score_map, roi_mask=leaf_mask)` — chỉ thống kê
  histogram TRONG phiến lá để nền không làm lệch ngưỡng.

QUY TẮC BẮT BUỘC (CODE FROM SCRATCH):
- TUYỆT ĐỐI KHÔNG gọi `cv2.threshold(..., cv2.THRESH_OTSU)` hoặc `cv2.calcHist`.
- Thuật toán được tính toán dựa trên vector hóa mảng 1D bằng NumPy (`np.cumsum`), không có vòng lặp
  Python theo ngưỡng t.

Tương thích OpenCV:
- Quy ước ngưỡng giống `cv2.THRESH_BINARY`: pixel > T* -> 255, pixel <= T* -> 0.
- Khi đường cong σ²_B có nhiều cực đại bằng nhau (histogram có "khoảng trống" giữa 2 mode),
  chọn t NHỎ NHẤT đạt cực đại — giống OpenCV (so sánh `>` nghiêm ngặt). Để kết quả không bị
  ảnh hưởng bởi sai số làm tròn dấu phẩy động, các giá trị chênh với cực đại dưới 1e-9 (tương đối)
  được coi là bằng nhau.
"""

import time
from typing import Any, Dict, Optional
import numpy as np

_NUM_LEVELS = 256
_TIE_RTOL = 1e-9


def _as_uint8_gray(gray_img: np.ndarray, func_name: str) -> np.ndarray:
    """Kiểm tra ảnh 2D và đưa về uint8 an toàn (làm tròn + clip nếu là float/int rộng)."""
    if not isinstance(gray_img, np.ndarray):
        raise ValueError(f"{func_name}: gray_img phải là np.ndarray, nhận được {type(gray_img).__name__}.")
    if gray_img.ndim != 2:
        raise ValueError(f"{func_name}: gray_img phải là mảng 2 chiều (H, W), nhận được {gray_img.shape}.")
    if gray_img.dtype == np.uint8:
        return gray_img
    if gray_img.dtype == np.bool_:
        return gray_img.astype(np.uint8) * 255
    arr = gray_img.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{func_name}: gray_img chứa NaN/inf.")
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def _roi_bool(roi_mask: Optional[np.ndarray], shape: tuple, func_name: str) -> Optional[np.ndarray]:
    if roi_mask is None:
        return None
    roi_mask = np.asarray(roi_mask)
    if roi_mask.ndim == 3 and roi_mask.shape[2] == 1:
        roi_mask = roi_mask[:, :, 0]
    if roi_mask.shape != shape:
        raise ValueError(f"{func_name}: roi_mask {roi_mask.shape} không khớp kích thước ảnh {shape}.")
    return roi_mask > 0


def compute_histogram_scratch(
    gray_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tính biểu đồ phân bố mức xám (Histogram) 256 mức độ sáng từ đầu bằng NumPy.

    Args:
        gray_img: Mảng ảnh mức xám 2 chiều kích thước `(H, W)`, kiểu dữ liệu `np.uint8`,
            các giá trị nằm trong đoạn `[0, 255]`. Mảng float/int khác sẽ được làm tròn và clip.
        roi_mask: (Tùy chọn) Mặt nạ vùng quan tâm 2 chiều kích thước `(H, W)`.
            Nếu được truyền vào, chỉ những pixel có `roi_mask > 0` mới được đưa vào thống kê.

    Returns:
        np.ndarray: Mảng 1 chiều kích thước `(256,)`, kiểu dữ liệu `np.int64`, trong đó
            phần tử tại vị trí `i` biểu diễn số lượng pixel có mức xám bằng `i`.

    Raises:
        ValueError: Nếu `gray_img` không phải mảng 2 chiều hoặc `roi_mask` không khớp kích thước.

    Note:
        - QUY TẮC: Không dùng `cv2.calcHist`.
        - Đếm tần suất bằng `np.bincount` (đếm số lần xuất hiện của mỗi số nguyên) trên mảng đã làm phẳng.
    """
    gray_u8 = _as_uint8_gray(gray_img, "compute_histogram_scratch")
    valid = _roi_bool(roi_mask, gray_u8.shape, "compute_histogram_scratch")
    pixels = gray_u8[valid] if valid is not None else gray_u8.ravel()
    return np.bincount(pixels, minlength=_NUM_LEVELS).astype(np.int64)


def compute_between_class_variance(hist: np.ndarray) -> np.ndarray:
    """Tính đường cong phương sai giữa hai lớp σ²_B(t) cho mọi t ∈ [0, 255] từ histogram.

    σ²_B(t) = (μ_T · ω0(t) − μ(t))² / (ω0(t) · (1 − ω0(t)))
    với ω0(t) = Σ_{i<=t} p_i, μ(t) = Σ_{i<=t} i·p_i, μ_T = μ(255).
    Tại các t mà ω0 = 0 hoặc ω1 = 0 (một lớp rỗng) gán σ²_B = 0.

    Args:
        hist: Histogram `(256,)` số nguyên không âm.

    Returns:
        np.ndarray: `(256,)` kiểu `np.float64`.
    """
    hist = np.asarray(hist, dtype=np.float64)
    if hist.shape != (_NUM_LEVELS,):
        raise ValueError(f"hist phải có shape (256,), nhận được {hist.shape}.")
    total = hist.sum()
    if total <= 0:
        return np.zeros(_NUM_LEVELS, dtype=np.float64)

    prob = hist / total
    levels = np.arange(_NUM_LEVELS, dtype=np.float64)
    omega0 = np.cumsum(prob)           # ω0(t)
    mu_cum = np.cumsum(levels * prob)  # μ(t)
    mu_total = mu_cum[-1]              # μ_T
    omega1 = 1.0 - omega0              # ω1(t)

    denom = omega0 * omega1
    variance = np.zeros(_NUM_LEVELS, dtype=np.float64)
    valid = denom > 1e-12
    variance[valid] = (mu_total * omega0[valid] - mu_cum[valid]) ** 2 / denom[valid]
    return variance


def otsu_threshold_scratch(
    gray_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    invert: bool = False,
) -> Dict[str, Any]:
    """Tìm ngưỡng nhị phân hóa tối ưu Otsu và tạo mặt nạ nhị phân từ đầu bằng NumPy.

    Nguyên lý toán học của phương pháp Otsu:
    Giả sử ảnh có L = 256 mức xám [0, 1, ..., L-1].
    Xác suất xuất hiện mức xám i: p_i = hist[i] / N, với N là tổng số pixel.
    Với mỗi ngưỡng thử nghiệm t in [0, 255]:
        - Xác suất lớp C0 (nền) và C1 (tiền cảnh):
            omega_0(t) = sum_{i=0}^t p_i
            omega_1(t) = 1 - omega_0(t)
        - Giá trị kỳ vọng trung bình lớp C0 và C1:
            mu_0(t) = sum_{i=0}^t (i * p_i) / omega_0(t)
            mu_1(t) = sum_{i=t+1}^{L-1} (i * p_i) / omega_1(t)
        - Phương sai giữa hai lớp (Between-class variance):
            sigma_B^2(t) = omega_0(t) * omega_1(t) * (mu_0(t) - mu_1(t))^2
                         = (mu_T * omega_0(t) - mu(t))^2 / (omega_0(t) * (1 - omega_0(t)))
    Ngưỡng tối ưu T* là giá trị t tối đa hóa sigma_B^2(t):
        T* = argmax_{0 <= t < L} sigma_B^2(t)

    Args:
        gray_img: Mảng ảnh mức xám 2 chiều kích thước `(H, W)`, kiểu `np.uint8`.
        roi_mask: (Tùy chọn) Mặt nạ vùng quan tâm kích thước `(H, W)`, kiểu `np.uint8` hoặc `bool`.
            Histogram chỉ tính trong ROI và pixel ngoài ROI luôn = 0 trong `binary_mask`.
        invert: (Tùy chọn) Nếu True, tiền cảnh là phần TỐI (pixel <= T* -> 255). Dùng khi đối tượng
            tối hơn nền (vd. lá xanh đậm trên giấy trắng khi phân ngưỡng ảnh xám). Mặc định False.

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa đầy đủ các khóa kết quả sau:
            - "threshold": int, giá trị ngưỡng tối ưu T* tìm được (0 <= T* <= 255).
            - "binary_mask": np.ndarray kích thước `(H, W)`, kiểu `np.uint8`,
              chỉ nhận hai giá trị duy nhất {0, 255}. Mặc định pixel > T* là 255 (tiền cảnh).
            - "histogram": np.ndarray kích thước `(256,)`, kiểu `np.int64`, biểu đồ tần suất mức xám.
            - "variance_curve": np.ndarray kích thước `(256,)`, kiểu `np.float64`,
              đường cong phương sai giữa các lớp sigma_B^2(t) tại mọi giá trị t.
            - "exec_time_ms": float, thời gian thực thi thuật toán tính bằng mili-giây (ms).
            - "separability": float trong [0, 1], η = σ²_B(T*) / σ²_T — độ tách lớp
              (gần 1: histogram 2 đỉnh rõ, Otsu đáng tin; gần 0: ảnh gần đơn mode).

    Raises:
        ValueError: Nếu kích thước mảng không hợp lệ hoặc `roi_mask` không khớp kích thước.

    Note:
        - QUY TẮC: Tuyệt đối không dùng `cv2.threshold`.
        - Ảnh hoàn toàn đồng nhất (hoặc ROI rỗng) KHÔNG gây lỗi: σ²_B ≡ 0, trả về T* = 0 giống
          hành vi của OpenCV, để pipeline không bị dừng giữa chừng.
        - Chia cho 0 khi omega_0(t) = 0 hoặc omega_1(t) = 0 được xử lý bằng cách gán sigma_B^2(t) = 0.
    """
    t_start = time.perf_counter()
    gray_u8 = _as_uint8_gray(gray_img, "otsu_threshold_scratch")
    valid = _roi_bool(roi_mask, gray_u8.shape, "otsu_threshold_scratch")

    hist = compute_histogram_scratch(gray_u8, roi_mask=valid)
    variance_curve = compute_between_class_variance(hist)

    max_var = float(variance_curve.max())
    if max_var > 0.0:
        # t nhỏ nhất đạt cực đại (chống nhiễu so sánh dấu phẩy động trên vùng phẳng)
        threshold = int(np.flatnonzero(variance_curve >= max_var * (1.0 - _TIE_RTOL))[0])
    else:
        threshold = 0

    # Độ tách lớp η = σ²_B / σ²_T
    total = hist.sum()
    if total > 0:
        levels = np.arange(_NUM_LEVELS, dtype=np.float64)
        prob = hist / total
        mu_t = float(np.sum(levels * prob))
        var_t = float(np.sum(((levels - mu_t) ** 2) * prob))
        separability = max_var / var_t if var_t > 0 else 0.0
    else:
        separability = 0.0

    foreground = gray_u8 <= threshold if invert else gray_u8 > threshold
    if valid is not None:
        foreground &= valid
    binary_mask = np.where(foreground, 255, 0).astype(np.uint8)

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "threshold": threshold,
        "binary_mask": binary_mask,
        "histogram": hist,
        "variance_curve": variance_curve,
        "exec_time_ms": float(max(exec_time_ms, 1e-6)),
        "separability": float(min(1.0, max(0.0, separability))),
    }
