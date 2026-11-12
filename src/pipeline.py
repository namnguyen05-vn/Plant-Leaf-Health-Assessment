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
from typing import Any, Dict, List, Optional, Union
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
    erode_scratch,
    fill_holes_bfs_scratch,
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


def _fallback_extract_disease_and_healthy_masks(
    rgb_img: np.ndarray,
    labels_2d: np.ndarray,
    centroids: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    yellow_halo_sensitivity: float = 0.65,
) -> Dict[str, Any]:
    """Fallback phân tích cụm dựa trên cả Tâm nâu (Necrosis) và Quầng vàng (Chlorosis)."""
    H, W = labels_2d.shape
    valid_labels = set(np.unique(labels_2d).tolist())
    valid_cluster_ids = sorted([int(cid) for cid in valid_labels if cid >= 0])

    if not valid_cluster_ids:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {"disease_mask": empty, "healthy_mask": empty.copy(), "disease_cluster_ids": []}

    sens = float(np.clip(yellow_halo_sensitivity, 0.0, 1.0))
    halo_rg_ratio = 0.94 - 0.18 * sens
    necrotic_rg_ratio = 0.90 - 0.08 * sens
    max_b_ratio = 0.58 + 0.18 * sens
    min_rg_minus_b = 30.0 - 18.0 * sens

    c_arr = centroids.astype(np.float32)
    exg_map: Dict[int, float] = {}
    rgb_map: Dict[int, tuple] = {}

    for cid in valid_cluster_ids:
        if c_arr.shape[1] >= 3:
            r, g, b = float(c_arr[cid, 0]), float(c_arr[cid, 1]), float(c_arr[cid, 2])
        else:
            r = g = b = float(c_arr[cid, 0])
        exg_map[cid] = 2.0 * g - r - b
        rgb_map[cid] = (r, g, b)

    # Ước lượng ExG trung vị của mô lá xanh (G > R) để làm mốc so sánh tâm cụm
    if leaf_mask is not None and int(np.sum(leaf_mask > 0)) > 0:
        leaf_rgb = rgb_img[leaf_mask > 0].astype(np.float32)
        green_pts = leaf_rgb[leaf_rgb[:, 1] > leaf_rgb[:, 0]]
        ref_pts = green_pts if green_pts.shape[0] > 0 else leaf_rgb
        leaf_exg_med = float(np.median(2.0 * ref_pts[:, 1] - ref_pts[:, 0] - ref_pts[:, 2]))
    else:
        leaf_exg_med = max(exg_map.values())

    disease_cluster_ids: List[int] = []
    for cid in valid_cluster_ids:
        r, g, b = rgb_map[cid]
        exg = exg_map[cid]
        bright = (r + g + b) / 3.0

        is_green_glare = (r > 190.0) and (g > 205.0) and (b > 175.0) and (g > r * 1.05)
        is_strong_green = (g > r * 1.26) and (exg > 0.45 * leaf_exg_med) and ((r - g) < -25.0)
        if is_green_glare or is_strong_green:
            continue

        # Điều kiện 1: Tâm nâu/đen hoại tử (Necrotic)
        is_necrotic = (
            (r > g * necrotic_rg_ratio)
            or (exg < min(30.0, leaf_exg_med - (125.0 - 55.0 * sens)) and r >= g * 0.78 and bright < 175.0)
        )
        # Điều kiện 2: Quầng vàng / vàng cam (Chlorotic Halo)
        is_yellow_halo = (
            (r > g * halo_rg_ratio)
            and (g > b * (1.38 - 0.20 * sens))
            and (max(r, g) >= 70.0)
        )
        # Điều kiện 3: Tâm bệnh khô trắng / bạc màu (Bleached / Tan Center)
        is_bleached = (
            (exg < (0.38 + 0.11 * sens) * leaf_exg_med)
            and (r >= g * 0.85)
            and (g <= r * 1.05)
        )

        if is_necrotic or is_yellow_halo or is_bleached:
            disease_cluster_ids.append(cid)

    # Bảo vệ: giữ lại ít nhất cụm xanh khỏe nhất nếu tất cả đều bị gắn nhãn bệnh
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
    """Xây dựng Bản đồ đặc trưng thực vật xanh chuẩn (Green Vegetation Feature Map) bằng NumPy.

    Chiếc lá khỏe có màu xanh đặc trưng: `ExG = 2*G - R - B > 0` VÀ `G > R`.
    Phông nền màu vàng rực, da tay, gỗ/đất nâu đều có `R >= G` nên bị gán bằng 0 tuyệt đối,
    chống tràn `leaf_mask` ra 4 viền ảnh.

    Args:
        rgb_img: Ảnh RGB `(H, W, 3)` `uint8`.
        hsv_img: Ảnh HSV `(H, W, 3)` `uint8` (kênh S ở trục 1).

    Returns:
        np.ndarray: Bản đồ đặc trưng 2D `(H, W)` kiểu `np.uint8`.
    """
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)

    exg_pos = np.clip(2.0 * g - r - b, 0.0, 255.0)
    sat = hsv_img[:, :, 1].astype(np.float32)

    # Chỉ nhận mô thực vật xanh thực sự (G - R >= 3 và ExG >= 8), loại bỏ phông vàng/nâu/xám
    is_green_veg = ((g - r) >= 3.0) & (exg_pos >= 8.0)
    veg_raw = np.sqrt(exg_pos * sat)
    veg_lifted = np.where(is_green_veg, np.clip(55.0 + 0.78 * veg_raw, 0.0, 255.0), 0.0)
    return np.clip(np.round(veg_lifted), 0, 255).astype(np.uint8)


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
    """Kết hợp cụm K-means, Otsu trên bản đồ Disease_Score và 3 dạng bệnh lý bên trong leaf_mask.

    Bắt trọn đồng thời 3 dạng tổn thương bên trong `leaf_mask`:
        1. Hoại tử nâu/đen (Necrotic): `R > G * 0.90` hoặc độ sáng tối bất thường so với mô lá xanh.
        2. Quầng vàng (Chlorotic Halo): `R > G * 0.82` và `G > B * 1.25`.
        3. Tâm bệnh khô trắng/bạc màu (Bleached/Tan Center): `ExG < 0.45 * median_ExG` và `R >= G * 0.85`.
    """
    leaf_bool = leaf_mask > 0
    if int(np.sum(leaf_bool)) == 0:
        empty = np.zeros_like(leaf_mask, dtype=np.uint8)
        return {"disease_mask": empty, "healthy_mask": empty.copy(), "rg_otsu_threshold": 0}

    sens = float(np.clip(yellow_halo_sensitivity, 0.0, 1.0))

    r = blurred_rgb[:, :, 0].astype(np.float32)
    g = blurred_rgb[:, :, 1].astype(np.float32)
    b = blurred_rgb[:, :, 2].astype(np.float32)

    exg = 2.0 * g - r - b
    rg_diff = r - g
    brightness = (r + g + b) / 3.0

    # Tính mốc màu chuẩn của mô lá xanh khỏe (chỉ xét pixel có G > R bên trong leaf_mask)
    healthy_ref = leaf_bool & (g > r)
    if int(np.sum(healthy_ref)) == 0:
        healthy_ref = leaf_bool

    median_r = float(np.median(r[healthy_ref]))
    median_g = float(np.median(g[healthy_ref]))
    median_b = float(np.median(b[healthy_ref]))
    median_exg = float(np.median(exg[healthy_ref]))
    median_rg = float(np.median(rg_diff[healthy_ref]))
    healthy_bright = (median_r + median_g + median_b) / 3.0

    # Phân ngưỡng Otsu trên bản đồ độ lệch màu bệnh học Disease_Score bên trong leaf_mask
    exg_drop = np.clip(median_exg - exg, 0.0, 255.0)
    rg_shift_pos = np.clip(rg_diff - median_rg, 0.0, 255.0)
    disease_score = np.clip(np.round(0.55 * rg_shift_pos + 0.45 * exg_drop), 0.0, 255.0).astype(np.uint8)
    score_otsu_res = call_otsu(disease_score, roi_mask=leaf_mask, fallback_tracker=fallback_tracker)
    score_otsu_mask = score_otsu_res["binary_mask"] > 0

    # Các ngưỡng màu động theo thanh trượt Yellow Halo Sensitivity (0.0 .. 1.0)
    necrotic_ratio = 0.94 - 0.06 * sens         # ~0.90 tại sens=0.65
    halo_ratio = 0.94 - 0.18 * sens             # ~0.82 tại sens=0.65
    gb_ratio = 1.38 - 0.20 * sens               # ~1.25 tại sens=0.65
    bleached_exg_factor = 0.38 + 0.11 * sens    # ~0.45 tại sens=0.65

    # Bảo vệ vùng bóng nắng xanh chói (R, G, B cao nhưng G vẫn vượt trội R > 5%) và lá xanh khỏe
    is_green_glare = (r > 190.0) & (g > 205.0) & (b > 175.0) & (g > r * 1.05)
    is_healthy_green = (g > r * (1.28 - 0.04 * sens)) & (exg > 0.45 * median_exg) & (rg_diff < -25.0)

    # 1. Hoại tử nâu / đen (Necrotic)
    cond_necrotic = (r > g * necrotic_ratio) | (
        (brightness < max(45.0, 0.55 * healthy_bright))
        & (exg < 0.40 * median_exg)
        & (r >= g * 0.78)
    )

    # 2. Quầng vàng / vàng cam (Chlorotic Halo) bên trong leaf_mask
    cond_yellow_halo = (r > g * halo_ratio) & (g > b * gb_ratio)

    # 3. Tâm bệnh khô trắng / bạc màu (Bleached / Tan Center như ổ bệnh #29, #24)
    cond_bleached = (exg < bleached_exg_factor * median_exg) & (r >= g * 0.85) & (g <= r * 1.05)

    # 4. Kết hợp Otsu trên bản đồ độ lệch bệnh học Disease_Score
    cond_score_otsu = score_otsu_mask & (r >= g * 0.85) & (g <= r * 1.15)

    cluster_bool = cluster_disease_mask > 0
    raw_disease_bool = leaf_bool & (~is_green_glare) & (~is_healthy_green) & (
        cluster_bool | cond_necrotic | cond_yellow_halo | cond_bleached | cond_score_otsu
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
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    min_lesion_area: int = 20,
    morph_kernel: int = 5,
    yellow_halo_sensitivity: float = 0.65,
    run_benchmark_comparison: bool = True,
) -> Dict[str, Any]:
    """Thực thi toàn bộ quy trình chẩn đoán sức khỏe lá cây (nâng cấp chống nhiễu nền & bắt quầng vàng).

    Args:
        file_source: Đường dẫn file ảnh, mảng NumPy RGB `(H, W, 3)` hoặc `(H, W)`, hoặc đối tượng upload buffer.
        k_clusters: Số cụm K-means trong không gian đặc trưng bệnh học (mặc định: 4).
        blur_kernel: Kích thước kernel lọc Gauss làm mịn (mặc định: 5).
        blur_sigma: Độ lệch chuẩn sigma của kernel lọc Gauss (mặc định: 1.0).
        min_lesion_area: Ngưỡng diện tích điểm ảnh tối thiểu để công nhận ổ bệnh (mặc định: 20).
        morph_kernel: Kích thước kernel cho phép đóng Closing gom quầng vàng + tâm nâu (mặc định: 5).
        yellow_halo_sensitivity: Độ nhạy bao phủ quầng vàng xung quanh ổ bệnh từ `0.0` đến `1.0` (mặc định: 0.65).
        run_benchmark_comparison: Cờ kích hoạt chạy đối sánh định lượng với OpenCV (mặc định: True).

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa kết quả các tầng xử lý, mặt nạ, chẩn đoán y bạ và benchmark.
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
    hsv_img = prep["hsv"]
    otsu_channel = prep["otsu_input"]

    # 2. Giai đoạn 1: Tách khung xương lá xanh (core_green_leaf) -> Giữ lá chính tâm ảnh -> Closing vừa phải -> Lấp lỗ nội vùng BFS
    otsu_res = call_otsu(otsu_channel, fallback_tracker=fallback_modules)
    H, W = gray_img.shape
    core_green_leaf = otsu_res["binary_mask"]
    leaf_main_core = keep_largest_connected_component_scratch(core_green_leaf, prefer_central=True)
    roi_close_k = 7 if min(H, W) >= 128 else 5
    leaf_closed = closing_scratch(leaf_main_core, kernel_size=roi_close_k)
    leaf_mask = fill_holes_bfs_scratch(leaf_closed)

    if int(np.sum(leaf_mask > 0)) == 0:
        leaf_mask = np.full((H, W), 255, dtype=np.uint8)

    # 3. Giai đoạn 2: Phân cụm K-Means trên không gian đặc trưng nhạy bệnh học (6 kênh) bên trong leaf_mask
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

    # 4. Giai đoạn 3: Hình thái học Closing -> Lấp lỗ tâm bệnh bạc màu BFS -> Opening nhẹ -> Co mép gọt nhiễu viền răng cưa
    close_k = max(morph_kernel, 5)
    closed_disease = closing_scratch(color_res["disease_mask"], kernel_size=close_k)
    filled_disease = fill_holes_bfs_scratch(closed_disease)
    open_kernel = min(morph_kernel, 3)
    morphed_disease = opening_scratch(filled_disease, kernel_size=open_kernel)

    # Gọt bỏ lớp viền giao thoa 2px ở rìa mép lá để loại sạch 100% nhiễu viền răng cưa
    inner_leaf_mask = erode_scratch(leaf_mask, kernel_size=5)
    post_res = filter_margin_noise_components_scratch(
        morphed_disease,
        inner_mask=inner_leaf_mask,
        min_area=min_lesion_area,
        min_inner_ratio=0.40,
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
        "color_result": color_res,
        "morphed_disease_mask": morphed_disease,
        "postprocess_result": post_res,
        "annotated_img": annotated_img,
        "diagnosis": diag_res,
        "benchmark": benchmark_res,
        "using_fallback": len(fallback_modules) > 0,
        "fallback_modules": fallback_modules,
    }
