"""Full End-to-End Plant Leaf Health Diagnosis Pipeline Module.

Module điều phối luồng xử lý toàn trình (End-to-End Pipeline) kết nối tuần tự
tất cả các module thành phần của hệ thống phân tích sức khỏe lá cây:
    1. Đọc dữ liệu ảnh / ma trận CSV (`data_io`)
    2. Tiền xử lý: Làm mịn Gauss, chuyển đổi Grayscale và HSV -> Tách lá khỏi nền bằng Otsu trên kênh Saturation (`preprocess`, `otsu`)
    3. Phân cụm màu sắc mô lá bằng K-Means vector hóa trong vùng ROI (`kmeans`)
    4. Phân tích phổ màu: Tách biệt mô lá khỏe mạnh và mô hoại tử đốm bệnh (`color_clustering`)
    5. Hậu xử lý: Hình thái học nhị phân (Closing -> Opening) và thuật toán BFS đếm ổ bệnh (`postprocess`)
    6. Chẩn đoán y bạ cây trồng 5 cấp độ & Đối sánh định lượng với OpenCV (`benchmark`)

CƠ CHẾ FALLBACK THÔNG MINH:
- Khi gọi các hàm từ module của các thành viên khác (`data_io`, `preprocess`, `otsu`, `kmeans`, `color_clustering`),
  hệ thống bọc trong khối `try ... except NotImplementedError:`.
- Nếu module của thành viên đã được merge, pipeline chạy code thật của họ.
- Nếu hàm vẫn đang ném `NotImplementedError`, pipeline tự động kích hoạt hàm dự phòng nội bộ
  bằng NumPy và đánh dấu cờ `"using_fallback": True` để ứng dụng luôn vận hành trơn tru 100%.
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

from src.data_io import load_image_or_csv
from src.preprocess import (
    gaussian_filter_scratch,
    rgb_to_gray_scratch,
    rgb_to_hsv_scratch,
)
from src.otsu import otsu_threshold_scratch
from src.kmeans import kmeans_scratch
from src.color_clustering import extract_disease_and_healthy_masks
from src.postprocess import (
    closing_scratch,
    connected_components_bfs_scratch,
    draw_lesion_bboxes_scratch,
    opening_scratch,
)
from src.benchmark import (
    compute_image_similarity_metrics,
    compute_segmentation_metrics_scratch,
    diagnose_leaf_severity,
    run_opencv_kmeans,
    run_opencv_otsu,
)


# =============================================================================
# CÁC HÀM FALLBACK NỘI BỘ (CHỈ CHẠY KHI MODULE THÀNH VIÊN CHƯA MERGE)
# =============================================================================

def _fallback_load_image_or_csv(
    file_source: Union[str, Path, Any],
    max_size: int = 512,
) -> np.ndarray:
    """Fallback nội bộ đọc ảnh (.png/.jpg) hoặc ma trận (.csv) và resize giữ tỷ lệ."""
    if isinstance(file_source, np.ndarray):
        arr = file_source.astype(np.uint8)
        return arr

    # Xác định đuôi file từ đường dẫn hoặc đối tượng UploadedFile
    filename = ""
    if isinstance(file_source, (str, Path)):
        path_obj = Path(file_source)
        if not path_obj.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_source}")
        filename = path_obj.name.lower()
    elif hasattr(file_source, "name"):
        filename = str(file_source.name).lower()

    if filename.endswith(".csv"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        df = pd.read_csv(file_source, header=None)
        mat = np.clip(np.round(df.to_numpy(dtype=np.float64)), 0, 255).astype(np.uint8)
        return mat

    # Đọc ảnh thông qua PIL
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    pil_img = Image.open(file_source).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    return np.asarray(pil_img, dtype=np.uint8)


def _fallback_rgb_to_gray(rgb_img: np.ndarray) -> np.ndarray:
    """Fallback chuyển RGB sang Grayscale bằng công thức độ chói chuẩn."""
    if rgb_img.ndim == 2:
        return rgb_img.astype(np.uint8, copy=False)
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def _fallback_rgb_to_hsv(rgb_img: np.ndarray) -> np.ndarray:
    """Fallback chuyển RGB sang HSV chuẩn hóa [0..255] hoàn toàn bằng NumPy."""
    rgb = rgb_img.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    delta = cmax - cmin

    # Hue (0..360)
    h = np.zeros_like(cmax, dtype=np.float32)
    nonzero_delta = delta > 1e-6

    mask_r = nonzero_delta & (cmax == r)
    mask_g = nonzero_delta & (cmax == g)
    mask_b = nonzero_delta & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    # Saturation & Value
    s = np.where(cmax > 1e-6, delta / np.where(cmax > 1e-6, cmax, 1.0), 0.0)
    v = cmax

    h_u8 = np.clip(np.round((h / 360.0) * 255.0), 0, 255).astype(np.uint8)
    s_u8 = np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)
    v_u8 = np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)

    return np.stack([h_u8, s_u8, v_u8], axis=-1)


def _fallback_gaussian_filter(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Fallback lọc làm mịn Gauss 2D vector hóa bằng NumPy sliding_window_view."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        return img.astype(np.uint8, copy=False)

    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma ** 2)))
    kernel /= np.sum(kernel)

    if img.ndim == 2:
        padded = np.pad(img.astype(np.float64), half, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
        blurred = np.einsum("ijkl,kl->ij", windows, kernel)
        return np.clip(np.round(blurred), 0, 255).astype(np.uint8)

    C = img.shape[2]
    out = np.empty_like(img, dtype=np.uint8)
    for c in range(C):
        padded = np.pad(img[:, :, c].astype(np.float64), half, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
        blurred = np.einsum("ijkl,kl->ij", windows, kernel)
        out[:, :, c] = np.clip(np.round(blurred), 0, 255).astype(np.uint8)
    return out


def _fallback_otsu_threshold(
    gray_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fallback phân ngưỡng Otsu 256 mức xám vector hóa bằng NumPy."""
    t_start = time.perf_counter()
    gray_u8 = gray_img.astype(np.uint8, copy=False)

    if roi_mask is not None:
        valid = roi_mask > 0
        pixels = gray_u8[valid]
    else:
        valid = None
        pixels = gray_u8.ravel()

    if pixels.size == 0:
        exec_time_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "threshold": 0,
            "binary_mask": np.zeros_like(gray_u8, dtype=np.uint8),
            "histogram": np.zeros(256, dtype=np.int64),
            "variance_curve": np.zeros(256, dtype=np.float64),
            "exec_time_ms": float(exec_time_ms),
        }

    hist = np.bincount(pixels, minlength=256).astype(np.int64)
    prob = hist.astype(np.float64) / float(pixels.size)
    bins = np.arange(256, dtype=np.float64)

    omega_0 = np.cumsum(prob)
    omega_1 = 1.0 - omega_0
    mu_cum = np.cumsum(bins * prob)
    mu_total = mu_cum[-1]

    denom = omega_0 * omega_1
    variance_curve = np.zeros(256, dtype=np.float64)
    nz = denom > 1e-12
    variance_curve[nz] = ((mu_total * omega_0[nz] - mu_cum[nz]) ** 2) / denom[nz]

    threshold = int(np.argmax(variance_curve))
    if valid is not None:
        binary_mask = np.where(valid & (gray_u8 > threshold), 255, 0).astype(np.uint8)
    else:
        binary_mask = np.where(gray_u8 > threshold, 255, 0).astype(np.uint8)

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "threshold": threshold,
        "binary_mask": binary_mask,
        "histogram": hist,
        "variance_curve": variance_curve,
        "exec_time_ms": float(exec_time_ms),
    }


def _fallback_kmeans(
    img: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fallback K-Means vector hóa bằng NumPy với khởi tạo K-Means++."""
    t_start = time.perf_counter()
    H, W = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]

    if roi_mask is not None:
        valid_mask = roi_mask > 0
        X = img[valid_mask].reshape(-1, C).astype(np.float32)
    else:
        valid_mask = None
        X = img.reshape(-1, C).astype(np.float32)

    N = X.shape[0]
    if N == 0:
        exec_time_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "segmented_img": np.zeros_like(img, dtype=np.uint8),
            "labels_2d": np.full((H, W), -1, dtype=np.int32),
            "centroids": np.zeros((k, C), dtype=np.float32),
            "n_iter": 0,
            "exec_time_ms": float(exec_time_ms),
        }

    k_eff = min(k, N)
    rng = np.random.RandomState(random_state)
    centroids = np.zeros((k_eff, C), dtype=np.float32)
    centroids[0] = X[rng.randint(0, N)]
    min_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

    for i in range(1, k_eff):
        total_d = float(np.sum(min_dist_sq))
        if total_d <= 1e-12:
            idx = rng.randint(0, N)
        else:
            probs = min_dist_sq / total_d
            idx = int(rng.choice(N, p=probs))
        centroids[i] = X[idx]
        dist_new = np.sum((X - centroids[i]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_new)

    n_iter = 0
    labels = np.zeros(N, dtype=np.int32)
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)

    for it in range(1, max_iter + 1):
        n_iter = it
        c_sq = np.sum(centroids ** 2, axis=1)
        dist_sq = x_sq + c_sq - 2.0 * (X @ centroids.T)
        labels = np.argmin(dist_sq, axis=1).astype(np.int32)

        new_centroids = np.copy(centroids)
        for j in range(k_eff):
            pts = X[labels == j]
            if pts.shape[0] > 0:
                new_centroids[j] = np.mean(pts, axis=0)
            else:
                new_centroids[j] = X[rng.randint(0, N)]

        shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
        centroids = new_centroids
        if shift < tol:
            break

    full_centroids = np.zeros((k, C), dtype=np.float32)
    full_centroids[:k_eff] = centroids
    centers_u8 = np.clip(np.round(full_centroids), 0, 255).astype(np.uint8)

    if valid_mask is not None:
        labels_2d = np.full((H, W), -1, dtype=np.int32)
        labels_2d[valid_mask] = labels
        if img.ndim == 2:
            segmented_img = np.zeros((H, W), dtype=np.uint8)
            segmented_img[valid_mask] = centers_u8[labels, 0]
        else:
            segmented_img = np.zeros((H, W, C), dtype=np.uint8)
            segmented_img[valid_mask] = centers_u8[labels]
    else:
        labels_2d = labels.reshape(H, W)
        if img.ndim == 2:
            segmented_img = centers_u8[labels, 0].reshape(H, W)
        else:
            segmented_img = centers_u8[labels].reshape(H, W, C)

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "segmented_img": segmented_img,
        "labels_2d": labels_2d,
        "centroids": full_centroids,
        "n_iter": int(n_iter),
        "exec_time_ms": float(exec_time_ms),
    }


def _fallback_extract_disease_and_healthy_masks(
    rgb_img: np.ndarray,
    labels_2d: np.ndarray,
    centroids: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fallback phân tích màu tâm cụm dựa trên Excess Green Index (ExG = 2G - R - B)."""
    H, W = labels_2d.shape
    valid_labels = set(np.unique(labels_2d).tolist())
    valid_cluster_ids = sorted([int(cid) for cid in valid_labels if cid >= 0])

    if not valid_cluster_ids:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {"disease_mask": empty, "healthy_mask": empty.copy(), "disease_cluster_ids": []}

    # Tính ExG = 2G - R - B cho các tâm cụm hợp lệ
    c_arr = centroids.astype(np.float32)
    exg_map: Dict[int, float] = {}
    gr_diff_map: Dict[int, float] = {}
    brightness_map: Dict[int, float] = {}

    for cid in valid_cluster_ids:
        if c_arr.shape[1] >= 3:
            r, g, b = float(c_arr[cid, 0]), float(c_arr[cid, 1]), float(c_arr[cid, 2])
        else:
            r = g = b = float(c_arr[cid, 0])
        exg_map[cid] = 2.0 * g - r - b
        gr_diff_map[cid] = g - r
        brightness_map[cid] = (r + g + b) / 3.0

    best_exg = max(exg_map.values())
    disease_cluster_ids: List[int] = []

    for cid in valid_cluster_ids:
        exg = exg_map[cid]
        gr_diff = gr_diff_map[cid]
        bright = brightness_map[cid]
        # Cụm bệnh: sắc đỏ >= sắc xanh (nâu/vàng/cam) hoặc độ xanh giảm sâu / đốm hoại tử tối
        is_non_green = gr_diff <= 5.0
        is_low_exg = (exg < 0.35 * best_exg) and (exg < 85.0)
        is_dark_necrosis = (bright < 55.0) and (exg < 25.0)
        if is_non_green or is_low_exg or is_dark_necrosis:
            disease_cluster_ids.append(cid)

    # Bảo vệ: nếu mọi cụm đều bị đánh là bệnh, giữ lại cụm có ExG cao nhất làm mô khỏe
    if len(disease_cluster_ids) == len(valid_cluster_ids) and len(valid_cluster_ids) > 1:
        healthiest_cid = max(valid_cluster_ids, key=lambda c: exg_map[c])
        disease_cluster_ids = [c for c in disease_cluster_ids if c != healthiest_cid]

    disease_mask = np.zeros((H, W), dtype=np.uint8)
    healthy_mask = np.zeros((H, W), dtype=np.uint8)

    if disease_cluster_ids:
        disease_mask[np.isin(labels_2d, disease_cluster_ids)] = 255

    healthy_cluster_ids = [c for c in valid_cluster_ids if c not in disease_cluster_ids]
    if healthy_cluster_ids:
        healthy_mask[np.isin(labels_2d, healthy_cluster_ids)] = 255

    if leaf_mask is not None:
        roi_bool = leaf_mask > 0
        disease_mask = np.where(roi_bool, disease_mask, 0).astype(np.uint8)
        healthy_mask = np.where(roi_bool, healthy_mask, 0).astype(np.uint8)

    return {
        "disease_mask": disease_mask,
        "healthy_mask": healthy_mask,
        "disease_cluster_ids": disease_cluster_ids,
    }


# =============================================================================
# CÁC HÀM ĐIỀU PHỐI GỌI MODULE THÀNH VIÊN CÓ CƠ CHẾ FALLBACK
# =============================================================================

def call_load_image(
    file_source: Union[str, Path, np.ndarray, Any],
    max_size: int = 512,
    fallback_tracker: Optional[List[str]] = None,
) -> np.ndarray:
    """Đọc ảnh hoặc ma trận CSV, tự động dùng fallback nếu `data_io.py` chưa cài đặt."""
    if isinstance(file_source, np.ndarray):
        return file_source.astype(np.uint8, copy=False)
    try:
        return load_image_or_csv(file_source, max_size=max_size)
    except NotImplementedError:
        if fallback_tracker is not None and "data_io" not in fallback_tracker:
            fallback_tracker.append("data_io")
        return _fallback_load_image_or_csv(file_source, max_size=max_size)


def call_preprocess(
    img: np.ndarray,
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    fallback_tracker: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Thực thi tiền xử lý (RGB -> Gray, RGB -> HSV, Gaussian Blur) kèm cơ chế fallback."""
    if img.ndim == 2:
        rgb_img = np.stack([img, img, img], axis=-1).astype(np.uint8)
        is_gray_source = True
    else:
        rgb_img = img.astype(np.uint8, copy=False)
        is_gray_source = False

    try:
        blurred_rgb = gaussian_filter_scratch(rgb_img, kernel_size=blur_kernel, sigma=blur_sigma)
    except NotImplementedError:
        if fallback_tracker is not None and "preprocess" not in fallback_tracker:
            fallback_tracker.append("preprocess")
        blurred_rgb = _fallback_gaussian_filter(rgb_img, kernel_size=blur_kernel, sigma=blur_sigma)

    try:
        gray_img = rgb_to_gray_scratch(blurred_rgb)
    except NotImplementedError:
        if fallback_tracker is not None and "preprocess" not in fallback_tracker:
            fallback_tracker.append("preprocess")
        gray_img = _fallback_rgb_to_gray(blurred_rgb)

    try:
        hsv_img = rgb_to_hsv_scratch(blurred_rgb)
    except NotImplementedError:
        if fallback_tracker is not None and "preprocess" not in fallback_tracker:
            fallback_tracker.append("preprocess")
        hsv_img = _fallback_rgb_to_hsv(blurred_rgb)

    saturation = hsv_img[:, :, 1]
    # Nếu đầu vào là ảnh xám đơn kênh (độ bão hòa bằng 0 toàn ảnh), dùng kênh Grayscale cho Otsu
    otsu_input = gray_img if (is_gray_source or int(saturation.max()) == int(saturation.min())) else saturation

    return {
        "original_rgb": rgb_img,
        "blurred_rgb": blurred_rgb,
        "grayscale": gray_img,
        "hsv": hsv_img,
        "saturation": saturation,
        "otsu_input": otsu_input,
    }


def call_otsu(
    channel_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    fallback_tracker: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Gọi `otsu_threshold_scratch`, fallback sang cài đặt nội bộ nếu chưa merge."""
    try:
        return otsu_threshold_scratch(channel_img, roi_mask=roi_mask)
    except NotImplementedError:
        if fallback_tracker is not None and "otsu" not in fallback_tracker:
            fallback_tracker.append("otsu")
        return _fallback_otsu_threshold(channel_img, roi_mask=roi_mask)


def call_kmeans(
    img: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
    roi_mask: Optional[np.ndarray] = None,
    fallback_tracker: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Gọi `kmeans_scratch`, fallback sang cài đặt nội bộ nếu chưa merge."""
    try:
        return kmeans_scratch(
            img, k=k, max_iter=max_iter, tol=tol, random_state=random_state, roi_mask=roi_mask
        )
    except NotImplementedError:
        if fallback_tracker is not None and "kmeans" not in fallback_tracker:
            fallback_tracker.append("kmeans")
        return _fallback_kmeans(
            img, k=k, max_iter=max_iter, tol=tol, random_state=random_state, roi_mask=roi_mask
        )


def call_color_clustering(
    rgb_img: np.ndarray,
    labels_2d: np.ndarray,
    centroids: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    fallback_tracker: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Gọi `extract_disease_and_healthy_masks`, fallback sang cài đặt nội bộ nếu chưa merge."""
    try:
        return extract_disease_and_healthy_masks(
            rgb_img, labels_2d, centroids, leaf_mask=leaf_mask
        )
    except NotImplementedError:
        if fallback_tracker is not None and "color_clustering" not in fallback_tracker:
            fallback_tracker.append("color_clustering")
        return _fallback_extract_disease_and_healthy_masks(
            rgb_img, labels_2d, centroids, leaf_mask=leaf_mask
        )


# =============================================================================
# HÀM ĐIỀU PHỐI PIPELINE CHÍNH
# =============================================================================

def run_full_diagnosis_pipeline(
    file_source: Union[str, Path, np.ndarray, Any],
    k_clusters: int = 3,
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    min_lesion_area: int = 20,
    morph_kernel: int = 3,
    run_benchmark_comparison: bool = True,
) -> Dict[str, Any]:
    """Thực thi toàn bộ quy trình chẩn đoán sức khỏe lá cây từ ảnh đầu vào đến kết quả y bạ hoàn chỉnh.

    Luồng xử lý 4 giai đoạn:
        - Giai đoạn 1: Đọc dữ liệu -> Tiền xử lý làm mịn Gauss & chuyển đổi RGB/Gray/HSV
          -> Tách phiến lá khỏi nền bằng Otsu trên kênh độ bão hòa (Saturation).
        - Giai đoạn 2: Phân cụm màu sắc mô lá bằng K-Means trong vùng lá ROI &
          Phân tích màu tâm cụm để trích xuất mặt nạ vết bệnh và mô khỏe.
        - Giai đoạn 3: Hậu xử lý hình thái học (`closing_scratch`, `opening_scratch`)
          và đếm ổ bệnh + vẽ hộp bao Bounding Box bằng `connected_components_bfs_scratch`.
        - Giai đoạn 4: Chẩn đoán mức độ bệnh 5 cấp độ & Đối sánh định lượng với OpenCV.

    Args:
        file_source: Đường dẫn file ảnh, mảng NumPy RGB `(H, W, 3)` hoặc `(H, W)`, hoặc đối tượng upload buffer.
        k_clusters: Số cụm K-means phân đoạn màu mô lá (mặc định: 3).
        blur_kernel: Kích thước kernel lọc Gauss làm mịn (mặc định: 5).
        blur_sigma: Độ lệch chuẩn sigma của kernel lọc Gauss (mặc định: 1.0).
        min_lesion_area: Ngưỡng diện tích điểm ảnh tối thiểu để công nhận ổ bệnh (mặc định: 20).
        morph_kernel: Kích thước kernel cho phép đóng/mở hình thái học (mặc định: 3).
        run_benchmark_comparison: Cờ kích hoạt chạy đối sánh định lượng với OpenCV (mặc định: True).

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa toàn bộ kết quả các tầng xử lý trung gian,
            chẩn đoán y bạ, đối sánh OpenCV và cờ theo dõi fallback.
    """
    fallback_modules: List[str] = []

    # 1. Nạp ảnh & Tiền xử lý
    loaded_img = call_load_image(file_source, max_size=512, fallback_tracker=fallback_modules)
    prep = call_preprocess(
        loaded_img, blur_kernel=blur_kernel, blur_sigma=blur_sigma, fallback_tracker=fallback_modules
    )
    rgb_img = prep["original_rgb"]
    blurred_rgb = prep["blurred_rgb"]
    gray_img = prep["grayscale"]
    otsu_channel = prep["otsu_input"]

    # 2. Giai đoạn 1: Tách nền bằng Otsu trên kênh Saturation (hoặc Grayscale nếu đầu vào xám)
    otsu_res = call_otsu(otsu_channel, fallback_tracker=fallback_modules)
    # Làm mượt mặt nạ lá bằng closing để lấp lỗ hổng nhỏ do vết hoại tử tối màu
    leaf_mask = closing_scratch(otsu_res["binary_mask"], kernel_size=morph_kernel)

    # Bảo vệ: nếu không tìm thấy pixel lá nào (ví dụ ảnh đồng nhất), coi toàn ảnh là ROI
    if int(np.sum(leaf_mask > 0)) == 0:
        leaf_mask = np.full(gray_img.shape, 255, dtype=np.uint8)

    # 3. Giai đoạn 2: Phân cụm K-Means bên trong vùng lá ROI & Trích xuất vết bệnh
    kmeans_res = call_kmeans(
        blurred_rgb,
        k=k_clusters,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        roi_mask=leaf_mask,
        fallback_tracker=fallback_modules,
    )
    color_res = call_color_clustering(
        rgb_img,
        kmeans_res["labels_2d"],
        kmeans_res["centroids"],
        leaf_mask=leaf_mask,
        fallback_tracker=fallback_modules,
    )

    # 4. Giai đoạn 3: Hậu xử lý Hình thái học & BFS đếm ổ bệnh
    closed_disease = closing_scratch(color_res["disease_mask"], kernel_size=morph_kernel)
    morphed_disease = opening_scratch(closed_disease, kernel_size=morph_kernel)
    post_res = connected_components_bfs_scratch(morphed_disease, min_area=min_lesion_area)
    annotated_img = draw_lesion_bboxes_scratch(rgb_img, post_res["bboxes"])

    # 5. Giai đoạn 4: Chẩn đoán mức độ bệnh 5 cấp độ
    diag_res = diagnose_leaf_severity(
        leaf_mask=leaf_mask,
        disease_mask=post_res["cleaned_mask"],
        num_lesions=post_res["num_lesions"],
    )

    # 6. Đối sánh định lượng với OpenCV baseline (nếu được bật)
    benchmark_res: Dict[str, Any] = {}
    if run_benchmark_comparison:
        otsu_cv = run_opencv_otsu(otsu_channel)
        otsu_metrics = compute_segmentation_metrics_scratch(
            otsu_res["binary_mask"], otsu_cv["binary_mask"]
        )

        kmeans_cv = run_opencv_kmeans(
            blurred_rgb, k=k_clusters, max_iter=100, tol=1e-3, roi_mask=leaf_mask
        )
        kmeans_img_sim = compute_image_similarity_metrics(
            kmeans_res["segmented_img"], kmeans_cv["segmented_img"]
        )
        cv_color = call_color_clustering(
            rgb_img,
            kmeans_cv["labels_2d"],
            kmeans_cv["centroids"],
            leaf_mask=leaf_mask,
            fallback_tracker=fallback_modules,
        )
        kmeans_disease_metrics = compute_segmentation_metrics_scratch(
            color_res["disease_mask"], cv_color["disease_mask"]
        )

        benchmark_res = {
            "otsu_cv": otsu_cv,
            "otsu_metrics": otsu_metrics,
            "kmeans_cv": kmeans_cv,
            "kmeans_similarity": kmeans_img_sim,
            "kmeans_disease_metrics": kmeans_disease_metrics,
        }

    return {
        "original_rgb": rgb_img,
        "grayscale": gray_img,
        "hsv": prep["hsv"],
        "saturation": prep["saturation"],
        "blurred": blurred_rgb,
        "otsu_channel": otsu_channel,
        "otsu_result": otsu_res,
        "leaf_mask": leaf_mask,
        "kmeans_result": kmeans_res,
        "color_result": color_res,
        "morphed_disease_mask": morphed_disease,
        "postprocess_result": post_res,
        "annotated_img": annotated_img,
        "diagnosis": diag_res,
        "benchmark": benchmark_res,
        "using_fallback": len(fallback_modules) > 0,
        "fallback_modules": fallback_modules,
    }
