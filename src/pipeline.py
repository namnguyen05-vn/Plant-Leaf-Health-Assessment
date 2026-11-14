"""Full End-to-End Plant Leaf Health Diagnosis Pipeline Module.

Module điều phối luồng xử lý toàn trình (End-to-End Pipeline) kết nối tuần tự
tất cả các module thành phần của hệ thống phân tích sức khỏe lá cây (Phiên bản nâng cấp thực địa):
    1. Đọc dữ liệu ảnh / ma trận CSV (`data_io`)
    2. Tiền xử lý & Tách vùng lá ROI chống nhiễu nền đất/gỗ nâu (`preprocess`, `otsu`, `postprocess`):
       - Xây dựng Bản đồ đặc trưng thực vật kết hợp (Vegetation Feature Map = sqrt(clip(2G - R - B) * S))
       - Otsu Thresholding + Closing + Giữ thành phần liên thông lớn nhất + Lấp lỗ thủng BFS (Hole Filling)
    3. Phân đoạn bệnh học trong vùng lá (`kmeans`, `color_clustering`):
       - Không gian đặc trưng nhạy bệnh học (R, G, B, R-G, (R-G)/(R+G), Hue) với mặc định k=4
       - Kết hợp phân cụm K-means, phân ngưỡng Otsu trên bản đồ (R-G) và quy tắc màu Tâm nâu + Quầng vàng
         với tham số điều chỉnh độ nhạy `yellow_halo_sensitivity` (chống nhiễu chói nắng trắng)
    4. Hậu xử lý: Hình thái học (`closing_scratch` -> `opening_scratch`) gom các chấm nâu & quầng vàng
       thành ổ bệnh thống nhất + BFS đếm ổ bệnh & vẽ Bounding Box (`postprocess`)
    5. Chẩn đoán y bạ cây trồng 5 cấp độ & Đối sánh định lượng với OpenCV (`benchmark`)
"""

from pathlib import Path
import time
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
    dilate_scratch,
    draw_lesion_bboxes_scratch,
    erode_scratch,
    fill_holes_bfs_scratch,
    fill_small_holes_bfs_scratch,
    filter_margin_noise_components_scratch,
    keep_largest_connected_component_scratch,
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
        return file_source.astype(np.uint8)

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
        return np.clip(np.round(df.to_numpy(dtype=np.float64)), 0, 255).astype(np.uint8)

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

    h = np.zeros_like(cmax, dtype=np.float32)
    nonzero_delta = delta > 1e-6

    mask_r = nonzero_delta & (cmax == r)
    mask_g = nonzero_delta & (cmax == g)
    mask_b = nonzero_delta & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

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
    k: int = 4,
    max_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fallback K-Means vector hóa bằng NumPy với khởi tạo K-Means++ (hỗ trợ C kênh đặc trưng)."""
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


def rgb_to_lab_scratch(rgb_img: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh sRGB `(H, W, 3)` sang không gian CIE L*a*b* (D65) thuần NumPy.

    Đầu ra có hình dạng `(H, W, 3)` kiểu `np.float32`:
        - `L*` thuộc `[0, 100]`
        - `a*` khoảng `[-86, +98]` (âm = xanh lá, dương = đỏ/tím)
        - `b*` khoảng `[-108, +95]` (dương = vàng, âm = xanh dương)
    """
    rgb_f = np.clip(rgb_img.astype(np.float32) / 255.0, 0.0, 1.0)
    linear = np.where(
        rgb_f <= 0.04045,
        rgb_f / 12.92,
        np.power(np.clip((rgb_f + 0.055) / 1.055, 0.0, None), 2.4),
    )
    r, g, b = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]

    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    fx = np.where(x > eps, np.cbrt(np.clip(x, 0.0, None)), (kappa * x + 16.0) / 116.0)
    fy = np.where(y > eps, np.cbrt(np.clip(y, 0.0, None)), (kappa * y + 16.0) / 116.0)
    fz = np.where(z > eps, np.cbrt(np.clip(z, 0.0, None)), (kappa * z + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return np.stack([L, a, b_lab], axis=-1).astype(np.float32)


def _compute_leaf_healthy_baseline(
    rgb_img: np.ndarray,
    lab_img: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Ước lượng phân phối mô khỏe của chính chiếc lá (Median & MAD bền vững theo CIE Lab/ExG)."""
    H, W = rgb_img.shape[:2]
    leaf_bool = (leaf_mask > 0) if leaf_mask is not None else np.ones((H, W), dtype=bool)
    n_leaf = int(np.sum(leaf_bool))
    if n_leaf == 0:
        return {
            "leaf_bool": leaf_bool,
            "L": lab_img[:, :, 0],
            "a": lab_img[:, :, 1],
            "b": lab_img[:, :, 2],
            "exg": np.zeros((H, W), dtype=np.float32),
            "rg": np.zeros((H, W), dtype=np.float32),
            "base_L": 50.0, "base_a": -15.0, "base_b": 20.0, "base_exg": 40.0, "base_rg": -15.0,
            "sigma_L": 6.0, "sigma_a": 3.0, "sigma_b": 3.5, "sigma_exg": 12.0, "sigma_rg": 10.0,
        }

    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b_rgb = rgb_img[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b_rgb
    rg = r - g
    nrg = (r - g) / (r + g + 8.0)

    L_ch = lab_img[:, :, 0]
    a_ch = lab_img[:, :, 1]
    b_ch = lab_img[:, :, 2]

    # Chọn mô xanh khỏe đại diện trên chính lá; nếu > 92% lá bị vàng/cháy, lấy 20% vùng xanh nhất
    green_cands = leaf_bool & (a_ch <= -5.0) & (exg > 0.0)
    n_green = int(np.sum(green_cands))
    if n_green >= max(25, int(0.08 * n_leaf)):
        healthy_ref = green_cands
    elif n_green >= 20 and float(np.median(a_ch[leaf_bool])) > -4.0:
        healthy_ref = green_cands
    else:
        a_leaf = a_ch[leaf_bool]
        q25 = float(np.percentile(a_leaf, 25))
        healthy_ref = leaf_bool & (a_ch <= max(q25, -4.0))
        if int(np.sum(healthy_ref)) == 0:
            healthy_ref = leaf_bool

    def _med_mad(arr: np.ndarray, min_sig: float, max_sig: float) -> Tuple[float, float]:
        vals = arr[healthy_ref]
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        return med, float(np.clip(mad, min_sig, max_sig))

    base_L, sigma_L = _med_mad(L_ch, 4.0, 18.0)
    base_a, sigma_a = _med_mad(a_ch, 2.2, 12.0)
    base_b, sigma_b = _med_mad(b_ch, 2.5, 14.0)
    base_exg, sigma_exg = _med_mad(exg, 8.0, 40.0)
    base_rg, sigma_rg = _med_mad(rg, 6.0, 35.0)
    base_nrg, sigma_nrg = _med_mad(nrg, 0.025, 0.14)

    return {
        "leaf_bool": leaf_bool,
        "L": L_ch,
        "a": a_ch,
        "b": b_ch,
        "exg": exg,
        "rg": rg,
        "nrg": nrg,
        "base_L": base_L,
        "base_a": base_a,
        "base_b": base_b,
        "base_exg": base_exg,
        "base_rg": base_rg,
        "base_nrg": base_nrg,
        "sigma_L": sigma_L,
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
        "sigma_exg": sigma_exg,
        "sigma_rg": sigma_rg,
        "sigma_nrg": sigma_nrg,
    }


def _evaluate_adaptive_disease_conditions(
    stats: Dict[str, Any],
    sens: float,
) -> Dict[str, np.ndarray]:
    """Đánh giá đồng thời Z-score thích nghi và Cổng chênh lệch tuyệt đối tối thiểu."""
    L = stats["L"]
    a = stats["a"]
    b = stats["b"]
    exg = stats["exg"]
    rg = stats["rg"]
    nrg = stats["nrg"]

    da = a - stats["base_a"]
    db = b - stats["base_b"]
    d_exg = stats["base_exg"] - exg
    d_rg = rg - stats["base_rg"]
    d_nrg = nrg - stats["base_nrg"]
    d_L_dark = stats["base_L"] - L

    z_a = da / stats["sigma_a"]
    z_b = db / stats["sigma_b"]
    z_exg = d_exg / stats["sigma_exg"]
    z_nrg = d_nrg / stats["sigma_nrg"]
    z_dark = d_L_dark / stats["sigma_L"]

    z_thr = 3.2 - 1.0 * sens  # ~2.55 tại sens=0.65

    # Bảo vệ vùng bóng râm xanh khỏe (g - r vẫn vượt trội dù L* tối làm Lab a* co về 0)
    is_dark_green_shadow = (nrg <= -0.075) & (d_nrg < 0.08)

    # 1. Hoại tử Đỏ-Nâu / Gỉ sắt: vượt Z-score lẫn độ lệch tuyệt đối so với mô nền
    cond_red_brown = (
        (z_a >= z_thr)
        & (z_exg >= 0.70 * z_thr)
        & (z_nrg >= 0.80 * z_thr)
        & (da >= (11.0 - 2.0 * sens))
        & (d_exg >= (22.0 - 4.0 * sens))
        & (d_nrg >= (0.095 - 0.02 * sens))
        & (a > -5.5)
    )

    # 2. Cháy lá / Thối đen tối màu (Late Blight / Black Rot) - miễn nhiễm bóng râm xanh tối
    cond_dark_blight = (
        (z_dark >= z_thr)
        & (z_nrg >= 0.80 * z_thr)
        & (~is_dark_green_shadow)
        & (d_L_dark >= (14.0 - 3.0 * sens))
        & (d_nrg >= (0.095 - 0.02 * sens))
        & (da >= (5.5 - 1.0 * sens))
        & (L < max(46.0, stats["base_L"] - 10.0))
    )

    # 3. Quầng vàng bệnh lý (Chlorotic Halo): dịch vàng b* lẫn dịch đỏ-xanh da & R-G
    cond_yellow_halo = (
        (z_a >= 0.85 * z_thr)
        & (z_b >= 0.65 * z_thr)
        & (da >= (10.5 - 2.0 * sens))
        & (db >= (8.0 - 2.0 * sens))
        & (d_rg >= (26.0 - 5.0 * sens))
        & (b >= max(28.0, stats["base_b"] + 5.0 - 2.0 * sens))
    )

    # 4. Tâm bệnh khô trắng / bạc màu (Bleached / Tan Center)
    cond_bleached = (
        (z_a >= z_thr)
        & (z_exg >= z_thr)
        & (da >= (14.0 - 2.0 * sens))
        & (d_exg >= 35.0)
        & (a >= -3.5)
        & (rg >= -5.0)
    )

    # Bảo vệ vùng chói bóng nắng xanh khỏe & vùng mô xanh thuần
    is_green_glare = (L > max(76.0, stats["base_L"] + 18.0)) & (a < -4.5) & (rg < 0.0)
    is_healthy_green = ((a <= min(-8.0, stats["base_a"] + 5.0)) & (da < 7.0) & (db < 10.0)) | is_dark_green_shadow

    # Cổng tổng hợp nghiêm ngặt dùng cho K-means & Score Otsu để không bắt nhầm bóng nắng / gân lá khỏe
    has_min_pathology_shift = (~is_dark_green_shadow) & (
        ((z_a >= 0.85 * z_thr) & (da >= (10.5 - 2.0 * sens)) & (d_exg >= 20.0) & (d_nrg >= 0.085) & (a > -5.5))
        | ((z_dark >= 0.85 * z_thr) & (d_L_dark >= (14.0 - 2.5 * sens)) & (d_nrg >= 0.09) & (da >= 5.5))
        | ((z_a >= 0.80 * z_thr) & (da >= (10.5 - 2.0 * sens)) & (db >= (8.0 - 2.0 * sens)) & (b >= 28.0))
        | cond_bleached
    )

    any_direct = cond_red_brown | cond_dark_blight | cond_yellow_halo | cond_bleached
    return {
        "any_direct": any_direct,
        "has_min_pathology_shift": has_min_pathology_shift,
        "is_green_glare": is_green_glare,
        "is_healthy_green": is_healthy_green,
        "z_score_map": np.clip(
            14.0 * (
                np.clip(z_a, 0.0, None)
                + 0.55 * np.clip(z_b, 0.0, None)
                + 0.50 * np.clip(z_dark, 0.0, None)
                + 0.65 * np.clip(z_nrg, 0.0, None)
            ),
            0.0,
            255.0,
        ).astype(np.uint8),
    }


def _fallback_extract_disease_and_healthy_masks(
    rgb_img: np.ndarray,
    labels_2d: np.ndarray,
    centroids: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    yellow_halo_sensitivity: float = 0.65,
) -> Dict[str, Any]:
    """Fallback phân cụm bệnh lý thích nghi bằng Z-score/MAD trên CIE Lab của chính chiếc lá."""
    H, W = labels_2d.shape
    valid_labels = set(np.unique(labels_2d).tolist())
    valid_cluster_ids = sorted([int(cid) for cid in valid_labels if cid >= 0])

    if not valid_cluster_ids:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {"disease_mask": empty, "healthy_mask": empty.copy(), "disease_cluster_ids": []}

    sens = float(np.clip(yellow_halo_sensitivity, 0.0, 1.0))
    lab_img = rgb_to_lab_scratch(rgb_img)
    stats = _compute_leaf_healthy_baseline(rgb_img, lab_img, leaf_mask=leaf_mask)

    c_arr = centroids.astype(np.float32)
    c_rgb = np.zeros((len(valid_cluster_ids), 1, 3), dtype=np.uint8)
    for idx, cid in enumerate(valid_cluster_ids):
        if c_arr.shape[1] >= 3:
            c_rgb[idx, 0, :] = np.clip(np.round(c_arr[cid, :3]), 0, 255).astype(np.uint8)
        else:
            v = int(np.clip(round(float(c_arr[cid, 0])), 0, 255))
            c_rgb[idx, 0, :] = [v, v, v]
    c_lab = rgb_to_lab_scratch(c_rgb)[:, 0, :]

    z_thr = 3.0 - 0.9 * sens
    exg_map: Dict[int, float] = {}
    disease_cluster_ids: List[int] = []

    for idx, cid in enumerate(valid_cluster_ids):
        r, g, b = float(c_rgb[idx, 0, 0]), float(c_rgb[idx, 0, 1]), float(c_rgb[idx, 0, 2])
        Lc, ac, bc = float(c_lab[idx, 0]), float(c_lab[idx, 1]), float(c_lab[idx, 2])
        exg = 2.0 * g - r - b
        rg = r - g
        nrg = (r - g) / (r + g + 8.0)
        exg_map[cid] = exg

        da = ac - stats["base_a"]
        db = bc - stats["base_b"]
        d_exg = stats["base_exg"] - exg
        d_rg = rg - stats["base_rg"]
        d_nrg = nrg - stats["base_nrg"]
        d_L_dark = stats["base_L"] - Lc

        z_a = da / stats["sigma_a"]
        z_b = db / stats["sigma_b"]
        z_exg = d_exg / stats["sigma_exg"]
        z_nrg = d_nrg / stats["sigma_nrg"]
        z_dark = d_L_dark / stats["sigma_L"]

        if (Lc > max(76.0, stats["base_L"] + 18.0)) and (ac < -4.5) and (rg < 0.0):
            continue
        if ((ac <= min(-8.0, stats["base_a"] + 5.0)) and (da < 7.0) and (db < 10.0)) or (
            nrg <= -0.075 and d_nrg < 0.08
        ):
            continue

        is_necrotic = (
            (z_a >= z_thr)
            and (z_exg >= 0.65 * z_thr)
            and (z_nrg >= 0.75 * z_thr)
            and (da >= 10.5 - 2.0 * sens)
            and (d_exg >= 20.0)
            and (d_nrg >= 0.085)
            and (ac > -5.5)
        )
        is_dark = (
            (z_dark >= z_thr)
            and (z_nrg >= 0.75 * z_thr)
            and (d_L_dark >= 14.0 - 2.5 * sens)
            and (d_nrg >= 0.09)
            and (da >= 5.5 - 1.0 * sens)
            and (Lc < max(46.0, stats["base_L"] - 10.0))
        )
        is_halo = (
            (z_a >= 0.85 * z_thr)
            and (z_b >= 0.65 * z_thr)
            and (da >= 10.0 - 2.0 * sens)
            and (db >= 8.0 - 2.0 * sens)
            and (d_rg >= 25.0 - 5.0 * sens)
            and (bc >= max(28.0, stats["base_b"] + 5.0 - 2.0 * sens))
        )
        is_bleached = (
            (z_a >= z_thr)
            and (z_exg >= z_thr)
            and (da >= 13.5 - 2.0 * sens)
            and (d_exg >= 35.0)
            and (ac >= -3.5)
            and (rg >= -5.0)
        )
        if is_necrotic or is_dark or is_halo or is_bleached:
            disease_cluster_ids.append(cid)

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
# CÁC HÀM ĐẶC TRƯNG NÂNG CẤP TẦNG 1 & TẦNG 2 (NUMPY FROM SCRATCH)
# =============================================================================

def compute_vegetation_feature_map(
    rgb_img: np.ndarray,
    hsv_img: np.ndarray,
) -> np.ndarray:
    """Xây dựng Bản đồ tách tiền cảnh lá (Leaf Foreground Feature Map) bằng khoảng cách màu CIE Lab so với viền ảnh.

    Thay vì cắt cứng `G - R >= 3` (gây vỡ lá úa vàng và cụt ổ bệnh thối đen sát mép), hàm ước lượng
    tối đa 4 cụm màu đại diện từ dải viền ngoài khung hình trong không gian CIE L*a*b* rồi tính
    khoảng cách màu cảm nhận (nhấn mạnh sắc độ `a*, b*` gấp ~35 lần độ sáng `L*`) kết hợp khử bóng đổ
    xám trung tính trên nền phòng thí nghiệm và khuếch đại sắc xanh thực vật.
    """
    H, W = rgb_img.shape[:2]
    lab = rgb_to_lab_scratch(rgb_img)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    border_w = max(3, min(10, int(0.025 * min(H, W))))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:border_w, :] = True
    border_mask[-border_w:, :] = True
    border_mask[:, :border_w] = True
    border_mask[:, -border_w:] = True

    border_pts = lab[border_mask]
    n_border = border_pts.shape[0]
    if n_border > 2000:
        step = max(1, n_border // 2000)
        border_sample = border_pts[::step]
    else:
        border_sample = border_pts

    # Phân cụm nhẹ tối đa 4 màu nền ở dải viền bằng NumPy (xử lý đa phông nền như đất nâu + nền trắng/vàng)
    k_bg = min(4, border_sample.shape[0])
    idx_init = np.linspace(0, border_sample.shape[0] - 1, k_bg, dtype=int)
    centers = border_sample[idx_init].copy()
    bg_labels = np.zeros(border_sample.shape[0], dtype=np.int32)

    for _ in range(12):
        diffs = border_sample[:, None, :] - centers[None, :, :]
        sq_dists = (
            0.08 * (diffs[:, :, 0] ** 2)
            + 1.60 * (diffs[:, :, 1] ** 2)
            + 1.40 * (diffs[:, :, 2] ** 2)
        )
        new_labels = np.argmin(sq_dists, axis=1)
        if np.array_equal(new_labels, bg_labels) and _ > 0:
            break
        bg_labels = new_labels
        for j in range(k_bg):
            pts_j = border_sample[bg_labels == j]
            if pts_j.shape[0] > 0:
                centers[j] = np.mean(pts_j, axis=0)

    counts = np.bincount(bg_labels, minlength=k_bg)
    non_green_border_frac = float(np.mean(border_sample[:, 1] > -5.0))

    # Loại bỏ cụm nhiễu siêu nhỏ (<6% viền) hoặc ngọn/cuống lá xanh nhỏ lỡ chạm viền nền không-xanh
    valid_protos: List[np.ndarray] = []
    for j in range(k_bg):
        frac = float(counts[j]) / max(1, border_sample.shape[0])
        if frac < 0.06:
            continue
        if non_green_border_frac >= 0.65 and float(centers[j, 1]) < -6.0 and frac < 0.25:
            continue
        valid_protos.append(centers[j])

    if not valid_protos:
        best_j = int(np.argmax(counts))
        valid_protos = [centers[best_j]]

    proto_arr = np.stack(valid_protos, axis=0).astype(np.float32)  # (P, 3)

    # Tính khoảng cách màu tối thiểu tới các nguyên mẫu nền
    min_dist_sq = np.full((H, W), np.inf, dtype=np.float32)
    for j in range(proto_arr.shape[0]):
        d_sq = (
            0.04 * ((L - proto_arr[j, 0]) ** 2)
            + 1.60 * ((a - proto_arr[j, 1]) ** 2)
            + 1.40 * ((b - proto_arr[j, 2]) ** 2)
        )
        min_dist_sq = np.minimum(min_dist_sq, d_sq)

    min_dist = np.sqrt(np.clip(min_dist_sq, 0.0, None))

    # Khử bóng đổ xám/tối trung tính trên nền studio khi viền nền không phải nền vàng
    bg_med_b = float(np.median(border_sample[:, 2]))
    bg_med_a = float(np.median(border_sample[:, 1]))
    if bg_med_b < 14.0 and abs(bg_med_a) < 14.0:
        is_cast_shadow = (
            ((L < 30.0) & (a > -3.5) & (b < 4.5))
            | ((L < 34.0) & (a > -2.5) & (b < 8.0) & (np.hypot(a, b) < 9.5))
        )
        min_dist = np.where(is_cast_shadow, min_dist * 0.15, min_dist)

    # Nén mềm Michaelis-Menten theo tỷ lệ tương phản nền-lá (k_half=28): giữ ổ bệnh & vùng chói, gọt nhiễu rìa
    fg = np.where(min_dist >= 5.8, min_dist, 0.0)
    fg_scaled = np.where(fg > 0.0, 225.0 * (fg / (fg + 28.0)), 0.0)
    veg_boost = np.where(
        (min_dist >= 5.8) & (a < -3.0),
        np.clip((-a - 3.0) * 1.1, 0.0, 30.0),
        0.0,
    )
    return np.clip(np.round(fg_scaled + veg_boost), 0, 255).astype(np.uint8)


def build_pathology_feature_space(
    rgb_img: np.ndarray,
    hsv_img: np.ndarray,
) -> np.ndarray:
    """Chuyển đổi ảnh RGB sang không gian đặc trưng 6 kênh nhạy với bệnh lá cây.

    Các kênh đầu ra `(H, W, 6)` kiểu `np.uint8`:
        - Kênh 0..2: `R, G, B` gốc (giữ nguyên tương thích tọa độ màu tâm cụm)
        - Kênh 3: Độ lệch Đỏ-Xanh lá `clip(128 + 1.5 * (R - G), 0, 255)`
        - Kênh 4: Chỉ số vàng/hoại tử chuẩn hóa `clip(128 + 200 * (R - G) / (R + G + 1e-5), 0, 255)`
        - Kênh 5: Kênh sắc độ Hue `H` từ không gian HSV `[0, 255]`
    """
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)

    rg_diff = r - g
    norm_rg = rg_diff / (r + g + 1e-5)

    rg_u8 = np.clip(np.round(128.0 + 1.5 * rg_diff), 0.0, 255.0).astype(np.uint8)
    nrg_u8 = np.clip(np.round(128.0 + 200.0 * norm_rg), 0.0, 255.0).astype(np.uint8)
    hue_u8 = hsv_img[:, :, 0].astype(np.uint8)

    return np.stack(
        [rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2], rg_u8, nrg_u8, hue_u8],
        axis=-1,
    )


def detect_pathology_disease_mask(
    blurred_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    cluster_disease_mask: np.ndarray,
    yellow_halo_sensitivity: float = 0.65,
    fallback_tracker: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Phát hiện tổn thương thích nghi theo phân phối mô nền của từng lá (Adaptive Z-score / Robust MAD + Otsu).

    Bắt trọn đồng thời 4 dạng tổn thương bên trong `leaf_mask` mà không dùng ngưỡng tĩnh `R > G * 0.82/0.90`:
        1. Hoại tử Đỏ-Nâu / Gỉ sắt (Red-Brown / Rust Necrosis)
        2. Hoại tử Tối / Cháy lá (Dark Blight / Black Rot)
        3. Quầng vàng bệnh lý (Chlorotic Halo)
        4. Tâm bệnh khô trắng / bạc màu (Bleached / Tan Center)
    Chỉ xác nhận điểm bệnh khi thỏa đồng thời ngưỡng Z-score tương đối và Minimum Absolute Delta Gate.
    """
    leaf_bool = leaf_mask > 0
    if int(np.sum(leaf_bool)) == 0:
        empty = np.zeros_like(leaf_mask, dtype=np.uint8)
        return {"disease_mask": empty, "healthy_mask": empty.copy(), "rg_otsu_threshold": 0}

    sens = float(np.clip(yellow_halo_sensitivity, 0.0, 1.0))
    lab_img = rgb_to_lab_scratch(blurred_rgb)
    stats = _compute_leaf_healthy_baseline(blurred_rgb, lab_img, leaf_mask=leaf_mask)
    conds = _evaluate_adaptive_disease_conditions(stats, sens=sens)

    score_otsu_res = call_otsu(conds["z_score_map"], roi_mask=leaf_mask, fallback_tracker=fallback_tracker)
    score_otsu_mask = (score_otsu_res["binary_mask"] > 0) & conds["has_min_pathology_shift"]

    cluster_bool = (cluster_disease_mask > 0) & conds["has_min_pathology_shift"]
    raw_disease_bool = (
        leaf_bool
        & (~conds["is_green_glare"])
        & (~conds["is_healthy_green"])
        & (conds["any_direct"] | cluster_bool | score_otsu_mask)
    )

    disease_mask = np.where(raw_disease_bool, 255, 0).astype(np.uint8)
    healthy_mask = np.where(leaf_bool & (~raw_disease_bool), 255, 0).astype(np.uint8)

    return {
        "disease_mask": disease_mask,
        "healthy_mask": healthy_mask,
        "rg_otsu_threshold": int(score_otsu_res["threshold"]),
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
) -> Dict[str, Any]:
    """Thực thi tiền xử lý (Gauss Blur, RGB->Gray, RGB->HSV, Vegetation Feature Map) kèm fallback."""
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
    veg_map = compute_vegetation_feature_map(blurred_rgb, hsv_img)

    # Dùng Vegetation Map làm đầu vào Otsu tách lá khỏi nền đất/nâu; fallback Gray nếu ảnh đơn sắc
    if is_gray_source or (int(veg_map.max()) == int(veg_map.min())):
        otsu_input = gray_img
    else:
        otsu_input = veg_map

    return {
        "original_rgb": rgb_img,
        "blurred_rgb": blurred_rgb,
        "grayscale": gray_img,
        "hsv": hsv_img,
        "saturation": saturation,
        "vegetation_map": veg_map,
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
    k: int = 4,
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
    yellow_halo_sensitivity: float = 0.65,
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
            rgb_img,
            labels_2d,
            centroids,
            leaf_mask=leaf_mask,
            yellow_halo_sensitivity=yellow_halo_sensitivity,
        )


# =============================================================================
# HÀM ĐIỀU PHỐI PIPELINE CHÍNH
# =============================================================================

def run_full_diagnosis_pipeline(
    file_source: Union[str, Path, np.ndarray, Any],
    k_clusters: int = 4,
    blur_kernel: Optional[int] = None,
    blur_sigma: float = 1.0,
    min_lesion_area: Optional[int] = None,
    morph_kernel: Optional[int] = None,
    yellow_halo_sensitivity: float = 0.65,
    run_benchmark_comparison: bool = True,
) -> Dict[str, Any]:
    """Thực thi toàn bộ quy trình chẩn đoán sức khỏe lá cây (nâng cấp Lab Border + Adaptive Z-score + Small-Hole BFS).

    Args:
        file_source: Đường dẫn file ảnh, mảng NumPy RGB `(H, W, 3)` hoặc `(H, W)`, hoặc đối tượng upload buffer.
        k_clusters: Số cụm K-means trong không gian đặc trưng bệnh học (mặc định: 4).
        blur_kernel: Kích thước kernel Gauss; `None` tự chọn `3` cho ảnh `<= 320px` và `5` cho ảnh lớn hơn.
        blur_sigma: Độ lệch chuẩn sigma của kernel lọc Gauss (mặc định: 1.0).
        min_lesion_area: Ngưỡng diện tích ổ bệnh tối thiểu; `None` tự co giãn tỷ lệ kích thước ảnh (`~8 px` trên `256x256`).
        morph_kernel: Kích thước kernel Closing; `None` tự chọn `3` cho ảnh `<= 320px` và `5` cho ảnh lớn hơn.
        yellow_halo_sensitivity: Độ nhạy bao phủ quầng vàng xung quanh ổ bệnh từ `0.0` đến `1.0` (mặc định: 0.65).
        run_benchmark_comparison: Cờ kích hoạt chạy đối sánh định lượng với OpenCV (mặc định: True).

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa kết quả các tầng xử lý, mặt nạ, chẩn đoán y bạ và benchmark.
        cluster_color_result giữ nguyên kết quả H2 trước fusion (kể cả uncertain_mask
        và metadata); color_result vẫn là kết quả Lab/MAD fusion của pipeline.
    """
    fallback_modules: List[str] = []

    # 1. Nạp ảnh & Tự động co giãn siêu tham số theo độ phân giải thực tế
    loaded_img = call_load_image(file_source, max_size=512, fallback_tracker=fallback_modules)
    H_in, W_in = loaded_img.shape[:2]
    is_small_img = max(H_in, W_in) <= 320
    eff_blur_kernel = int(blur_kernel) if blur_kernel is not None else (3 if is_small_img else 5)
    eff_min_area = (
        int(min_lesion_area)
        if min_lesion_area is not None
        else max(6, min(20, int(round(0.00012 * H_in * W_in))))
    )
    eff_morph_kernel = int(morph_kernel) if morph_kernel is not None else (3 if is_small_img else 5)

    prep = call_preprocess(
        loaded_img, blur_kernel=eff_blur_kernel, blur_sigma=blur_sigma, fallback_tracker=fallback_modules
    )
    rgb_img = prep["original_rgb"]
    blurred_rgb = prep["blurred_rgb"]
    gray_img = prep["grayscale"]
    hsv_img = prep["hsv"]
    otsu_channel = prep["otsu_input"]

    # 2. Giai đoạn 1: Tách khung phiến lá (Border Lab Otsu) -> Closing -> Lấp lỗ nội vùng -> Cắt cuống lá mảnh (Petiole Removal)
    otsu_res = call_otsu(otsu_channel, fallback_tracker=fallback_modules)
    H, W = gray_img.shape
    core_green_leaf = otsu_res["binary_mask"]
    leaf_main_core = keep_largest_connected_component_scratch(core_green_leaf, prefer_central=True)
    roi_close_k = 7 if min(H, W) >= 128 else 5
    leaf_closed = closing_scratch(leaf_main_core, kernel_size=roi_close_k)
    leaf_filled = fill_holes_bfs_scratch(leaf_closed)

    leaf_area_pre = int(np.sum(leaf_filled > 0))
    petiole_k = 11 if min(H, W) <= 320 else 15
    if leaf_area_pre > max(400, int(0.04 * H * W)):
        blade_opened = opening_scratch(leaf_filled, kernel_size=petiole_k)
        blade_core = keep_largest_connected_component_scratch(blade_opened, prefer_central=True)
        if int(np.sum(blade_core > 0)) >= int(0.45 * leaf_area_pre):
            restore_k = 9 if min(H, W) <= 320 else 11
            blade_dilated = dilate_scratch(blade_core, kernel_size=restore_k)
            leaf_mask = np.where((blade_dilated > 0) & (leaf_filled > 0), 255, 0).astype(np.uint8)
        else:
            leaf_mask = leaf_filled
    else:
        leaf_mask = leaf_filled

    if int(np.sum(leaf_mask > 0)) == 0:
        leaf_mask = np.full((H, W), 255, dtype=np.uint8)

    # 3. Giai đoạn 2: Phân cụm K-Means + Sàng lọc Adaptive Z-score/MAD trên CIE Lab
    pathology_features = build_pathology_feature_space(blurred_rgb, hsv_img)
    kmeans_raw = call_kmeans(
        pathology_features,
        k=k_clusters,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        roi_mask=leaf_mask,
        fallback_tracker=fallback_modules,
    )
    centroids_rgb = kmeans_raw["centroids"][:, :3].astype(np.float32)
    segmented_rgb = kmeans_raw["segmented_img"][:, :, :3].astype(np.uint8)
    kmeans_res: Dict[str, Any] = {
        "segmented_img": segmented_rgb,
        "labels_2d": kmeans_raw["labels_2d"],
        "centroids": centroids_rgb,
        "n_iter": kmeans_raw["n_iter"],
        "exec_time_ms": kmeans_raw["exec_time_ms"],
    }

    cluster_color_res = call_color_clustering(
        blurred_rgb,
        kmeans_res["labels_2d"],
        centroids_rgb,
        leaf_mask=leaf_mask,
        yellow_halo_sensitivity=yellow_halo_sensitivity,
        fallback_tracker=fallback_modules,
    )
    pathology_res = detect_pathology_disease_mask(
        blurred_rgb=blurred_rgb,
        leaf_mask=leaf_mask,
        cluster_disease_mask=cluster_color_res["disease_mask"],
        yellow_halo_sensitivity=yellow_halo_sensitivity,
        fallback_tracker=fallback_modules,
    )
    color_res: Dict[str, Any] = {
        "disease_mask": pathology_res["disease_mask"],
        "healthy_mask": pathology_res["healthy_mask"],
        "disease_cluster_ids": cluster_color_res["disease_cluster_ids"],
        "rg_otsu_threshold": pathology_res["rg_otsu_threshold"],
    }

    # 4. Giai đoạn 3: Closing vừa tầm -> Lấp lỗ cục bộ giới hạn diện tích (<= 3% lá) -> Bảo toàn đốm nhỏ & gọt nhiễu viền
    leaf_area_px = max(1, int(np.sum(leaf_mask > 0)))
    close_k = max(eff_morph_kernel, 3)
    closed_disease = closing_scratch(color_res["disease_mask"], kernel_size=close_k)
    max_hole_area = max(30, int(round(0.03 * leaf_area_px)))
    filled_disease = fill_small_holes_bfs_scratch(closed_disease, max_hole_area=max_hole_area)

    if eff_min_area > 15:
        open_kernel = min(eff_morph_kernel, 3)
        morphed_disease = opening_scratch(filled_disease, kernel_size=open_kernel)
    else:
        morphed_disease = filled_disease

    # Gọt bỏ lớp viền giao thoa ở rìa mép lá nhưng giữ lại ổ bệnh sát mép lá thật sự
    margin_erode_k = 7 if eff_min_area > 15 else 5
    inner_leaf_mask = erode_scratch(leaf_mask, kernel_size=margin_erode_k)
    post_res = filter_margin_noise_components_scratch(
        morphed_disease,
        inner_mask=inner_leaf_mask,
        min_area=eff_min_area,
        min_inner_ratio=0.55,
    )
    annotated_img = draw_lesion_bboxes_scratch(rgb_img, post_res["bboxes"])

    # 5. Giai đoạn 4: Chẩn đoán mức độ bệnh 5 cấp độ
    diag_res = diagnose_leaf_severity(
        leaf_mask=leaf_mask,
        disease_mask=post_res["cleaned_mask"],
        num_lesions=post_res["num_lesions"],
    )

    # 6. Đối sánh định lượng với OpenCV baseline
    benchmark_res: Dict[str, Any] = {}
    if run_benchmark_comparison:
        otsu_cv = run_opencv_otsu(otsu_channel)
        otsu_metrics = compute_segmentation_metrics_scratch(
            otsu_res["binary_mask"], otsu_cv["binary_mask"]
        )

        kmeans_cv_raw = run_opencv_kmeans(
            pathology_features, k=k_clusters, max_iter=100, tol=1e-3, roi_mask=leaf_mask
        )
        kmeans_cv_rgb = {
            "segmented_img": kmeans_cv_raw["segmented_img"][:, :, :3].astype(np.uint8),
            "labels_2d": kmeans_cv_raw["labels_2d"],
            "centroids": kmeans_cv_raw["centroids"][:, :3].astype(np.float32),
            "exec_time_ms": kmeans_cv_raw["exec_time_ms"],
        }
        kmeans_img_sim = compute_image_similarity_metrics(
            kmeans_res["segmented_img"], kmeans_cv_rgb["segmented_img"]
        )
        cv_color = call_color_clustering(
            blurred_rgb,
            kmeans_cv_rgb["labels_2d"],
            kmeans_cv_rgb["centroids"],
            leaf_mask=leaf_mask,
            yellow_halo_sensitivity=yellow_halo_sensitivity,
            fallback_tracker=fallback_modules,
        )
        cv_pathology = detect_pathology_disease_mask(
            blurred_rgb=blurred_rgb,
            leaf_mask=leaf_mask,
            cluster_disease_mask=cv_color["disease_mask"],
            yellow_halo_sensitivity=yellow_halo_sensitivity,
            fallback_tracker=fallback_modules,
        )
        kmeans_disease_metrics = compute_segmentation_metrics_scratch(
            color_res["disease_mask"], cv_pathology["disease_mask"]
        )

        benchmark_res = {
            "otsu_cv": otsu_cv,
            "otsu_metrics": otsu_metrics,
            "kmeans_cv": kmeans_cv_rgb,
            "kmeans_similarity": kmeans_img_sim,
            "kmeans_disease_metrics": kmeans_disease_metrics,
        }

    return {
        "original_rgb": rgb_img,
        "grayscale": gray_img,
        "hsv": hsv_img,
        "saturation": prep["saturation"],
        "vegetation_map": prep["vegetation_map"],
        "blurred": blurred_rgb,
        "otsu_channel": otsu_channel,
        "otsu_result": otsu_res,
        "core_green_leaf": core_green_leaf,
        "leaf_mask": leaf_mask,
        "inner_leaf_mask": inner_leaf_mask,
        "kmeans_result": kmeans_res,
        "cluster_color_result": cluster_color_res,
        "color_result": color_res,
        "morphed_disease_mask": morphed_disease,
        "postprocess_result": post_res,
        "annotated_img": annotated_img,
        "diagnosis": diag_res,
        "benchmark": benchmark_res,
        "using_fallback": len(fallback_modules) > 0,
        "fallback_modules": fallback_modules,
    }
