"""Automated Data Contracts & Quality Verification Test Suite.

Bộ kiểm thử tự động kiểm tra tính toàn vẹn của chuẩn giao tiếp dữ liệu (Data Contracts):
- Kiểm tra kiểu dữ liệu (dtype): uint8, float32, int32, v.v.
- Kiểm tra kích thước ma trận (shape): (H, W), (H, W, 3), (256,), v.v.
- Kiểm tra khoảng giá trị hợp lệ (value range): [0..255], {0, 255}, [0.0..1.0].
- Kiểm tra đầy đủ các khóa từ điển (Dictionary Keys) theo quy ước.
- Kiểm tra khả năng xử lý an toàn và đo thời gian thực thi (Runtime).

Hướng dẫn dành cho Nhóm trưởng:
Chạy lệnh `pytest tests/test_contracts.py -v` khi pull nhánh của thành viên về.
Nếu module nào chưa hoàn thiện (còn NotImplementedError), test case sẽ tự động SKIP.
Khi thành viên nộp code, tất cả các assert trong test case tương ứng phải PASS 100%.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pytest

# Import các hàm từ src
from src.data_io import load_image_or_csv, matrix_to_csv_bytes, save_matrix_to_csv
from src.preprocess import (
    create_gaussian_kernel_2d,
    gaussian_filter_scratch,
    mean_filter_scratch,
    rgb_to_gray_scratch,
    rgb_to_hsv_scratch,
)
from src.otsu import compute_histogram_scratch, otsu_threshold_scratch
from src.kmeans import compute_euclidean_distances, kmeans_scratch
from src.color_clustering import compute_excess_green_index, extract_disease_and_healthy_masks
from src.postprocess import (
    closing_scratch,
    connected_components_bfs_scratch,
    dilate_scratch,
    erode_scratch,
    opening_scratch,
)
from src.benchmark import (
    compute_image_similarity_metrics,
    compute_segmentation_metrics_scratch,
    diagnose_leaf_severity,
    run_opencv_kmeans,
    run_opencv_otsu,
)
from src.pipeline import run_full_diagnosis_pipeline


# =============================================================================
# FIXTURES: Tạo dữ liệu giả lập (Synthetic Test Fixtures)
# =============================================================================

@pytest.fixture
def synthetic_leaf_rgb() -> np.ndarray:
    """Tạo ảnh lá cây giả lập kích thước 120x120 không gian màu RGB.

    - Nền đen (0, 0, 0)
    - Vùng lá cây xanh lục: hình tròn bán kính 40 tại tâm (R=30, G=180, B=40)
    - Vùng đốm bệnh nâu vàng: 2 đốm nhỏ bán kính 6 (R=160, G=110, B=30)
    """
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:120, :120]
    
    # Vùng phiến lá (tâm 60, 60, r=40)
    leaf_disk = (xx - 60) ** 2 + (yy - 60) ** 2 <= 40 ** 2
    img[leaf_disk] = [34, 139, 34]  # Forest Green

    # Đốm bệnh 1 (tâm 50, 50, r=7)
    spot1 = (xx - 50) ** 2 + (yy - 50) ** 2 <= 7 ** 2
    img[spot1] = [139, 69, 19]  # Saddle Brown

    # Đốm bệnh 2 (tâm 70, 65, r=5)
    spot2 = (xx - 70) ** 2 + (yy - 65) ** 2 <= 5 ** 2
    img[spot2] = [184, 134, 11]  # Dark Goldenrod

    return img


@pytest.fixture
def synthetic_leaf_gray(synthetic_leaf_rgb: np.ndarray) -> np.ndarray:
    """Tạo ảnh mức xám tương ứng từ ảnh giả lập."""
    return (
        0.299 * synthetic_leaf_rgb[:, :, 0]
        + 0.587 * synthetic_leaf_rgb[:, :, 1]
        + 0.114 * synthetic_leaf_rgb[:, :, 2]
    ).astype(np.uint8)


@pytest.fixture
def synthetic_leaf_mask() -> np.ndarray:
    """Tạo mặt nạ lá nhị phân chuẩn `{0, 255}` kích thước 120x120."""
    mask = np.zeros((120, 120), dtype=np.uint8)
    yy, xx = np.ogrid[:120, :120]
    mask[(xx - 60) ** 2 + (yy - 60) ** 2 <= 40 ** 2] = 255
    return mask


# =============================================================================
# 1. TEST DATA_IO CONTRACTS
# =============================================================================

def test_data_io_contract(synthetic_leaf_rgb: np.ndarray):
    """Kiểm tra hợp đồng I/O dữ liệu đọc/ghi ma trận và ảnh."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "test_matrix.csv"
        matrix_2d = synthetic_leaf_rgb[:, :, 0]

        # Kiểm tra save_matrix_to_csv
        try:
            save_matrix_to_csv(matrix_2d, csv_file)
            assert csv_file.exists(), "File CSV phải được tạo ra sau khi lưu."
        except NotImplementedError:
            pytest.skip("save_matrix_to_csv chưa được triển khai.")

        # Kiểm tra load_image_or_csv từ file CSV
        try:
            loaded_mat = load_image_or_csv(csv_file)
            assert isinstance(loaded_mat, np.ndarray), "Kết quả đọc phải là np.ndarray."
            assert loaded_mat.dtype == np.uint8, "Dtype từ CSV phải là np.uint8."
            assert loaded_mat.shape == matrix_2d.shape, "Shape đọc từ CSV không khớp."
        except NotImplementedError:
            pytest.skip("load_image_or_csv chưa được triển khai.")

        # Kiểm tra matrix_to_csv_bytes
        try:
            csv_bytes = matrix_to_csv_bytes(matrix_2d)
            assert isinstance(csv_bytes, bytes), "Kết quả phải là kiểu bytes."
            assert len(csv_bytes) > 0, "Dữ liệu bytes không được rỗng."
        except NotImplementedError:
            pytest.skip("matrix_to_csv_bytes chưa được triển khai.")


# =============================================================================
# 2. TEST PREPROCESS CONTRACTS
# =============================================================================

def test_preprocess_rgb_to_gray_contract(synthetic_leaf_rgb: np.ndarray):
    """Kiểm tra hợp đồng hàm rgb_to_gray_scratch."""
    try:
        gray = rgb_to_gray_scratch(synthetic_leaf_rgb)
    except NotImplementedError:
        pytest.skip("rgb_to_gray_scratch chưa được triển khai.")

    assert isinstance(gray, np.ndarray), "Đầu ra phải là np.ndarray."
    assert gray.ndim == 2, f"Ảnh xám phải là 2 chiều (H, W), thực tế: {gray.shape}."
    assert gray.shape == synthetic_leaf_rgb.shape[:2], "Kích thước (H, W) phải bảo toàn."
    assert gray.dtype == np.uint8, f"Kiểu dữ liệu phải là np.uint8, thực tế: {gray.dtype}."
    assert 0 <= gray.min() <= gray.max() <= 255, "Giá trị mức xám phải nằm trong [0, 255]."


def test_preprocess_rgb_to_hsv_contract(synthetic_leaf_rgb: np.ndarray):
    """Kiểm tra hợp đồng hàm rgb_to_hsv_scratch."""
    try:
        hsv = rgb_to_hsv_scratch(synthetic_leaf_rgb)
    except NotImplementedError:
        pytest.skip("rgb_to_hsv_scratch chưa được triển khai.")

    assert isinstance(hsv, np.ndarray), "Đầu ra phải là np.ndarray."
    assert hsv.shape == synthetic_leaf_rgb.shape, "HSV phải có cùng kích thước (H, W, 3)."
    assert hsv.dtype == np.uint8, f"Kiểu dữ liệu HSV phải là np.uint8, thực tế: {hsv.dtype}."
    assert 0 <= hsv.min() <= hsv.max() <= 255, "Giá trị các kênh HSV phải được chuẩn hóa về [0, 255]."


def test_preprocess_gaussian_filter_contract(synthetic_leaf_gray: np.ndarray):
    """Kiểm tra hợp đồng làm mịn Gaussian filter scratch."""
    try:
        kernel = create_gaussian_kernel_2d(kernel_size=5, sigma=1.0)
        assert kernel.shape == (5, 5), "Kernel Gauss phải có kích thước (5, 5)."
        assert np.isclose(np.sum(kernel), 1.0, atol=1e-4), "Tổng trọng số kernel Gauss phải bằng 1.0."

        blurred = gaussian_filter_scratch(synthetic_leaf_gray, kernel_size=5, sigma=1.0)
        assert blurred.shape == synthetic_leaf_gray.shape, "Ảnh làm mịn phải giữ nguyên kích thước."
        assert blurred.dtype == np.uint8, "Ảnh làm mịn phải có kiểu np.uint8."
    except NotImplementedError:
        pytest.skip("Gaussian filter scratch chưa được triển khai.")


def test_preprocess_mean_filter_contract(synthetic_leaf_gray: np.ndarray):
    """Kiểm tra hợp đồng bộ lọc trung bình Mean filter scratch."""
    try:
        blurred = mean_filter_scratch(synthetic_leaf_gray, kernel_size=3)
        assert blurred.shape == synthetic_leaf_gray.shape, "Ảnh sau lọc trung bình phải giữ nguyên shape."
        assert blurred.dtype == np.uint8, "Kiểu dữ liệu sau lọc trung bình phải là np.uint8."
    except NotImplementedError:
        pytest.skip("Mean filter scratch chưa được triển khai.")


# =============================================================================
# 3. TEST OTSU CONTRACTS
# =============================================================================

def test_otsu_histogram_contract(synthetic_leaf_gray: np.ndarray):
    """Kiểm tra hợp đồng hàm tính histogram mức xám 256 phần tử."""
    try:
        hist = compute_histogram_scratch(synthetic_leaf_gray)
    except NotImplementedError:
        pytest.skip("compute_histogram_scratch chưa được triển khai.")

    assert isinstance(hist, np.ndarray), "Histogram phải là np.ndarray."
    assert hist.shape == (256,), f"Histogram phải có đúng kích thước (256,), thực tế: {hist.shape}."
    assert hist.sum() == synthetic_leaf_gray.size, "Tổng số đếm histogram phải bằng tổng số pixel của ảnh."


def test_otsu_threshold_contract(synthetic_leaf_gray: np.ndarray):
    """Kiểm tra hợp đồng thuật toán phân ngưỡng tự động Otsu."""
    try:
        res = otsu_threshold_scratch(synthetic_leaf_gray)
    except NotImplementedError:
        pytest.skip("otsu_threshold_scratch chưa được triển khai.")

    # Kiểm tra đầy đủ các keys
    required_keys = {"threshold", "binary_mask", "histogram", "variance_curve", "exec_time_ms"}
    assert required_keys.issubset(res.keys()), f"Thiếu khóa trong kết quả Otsu: {required_keys - set(res.keys())}"

    # Kiểm tra ngưỡng T*
    assert isinstance(res["threshold"], (int, np.integer)), "Ngưỡng threshold phải là số nguyên int."
    assert 0 <= res["threshold"] <= 255, f"Ngưỡng T* phải nằm trong [0, 255], thực tế: {res['threshold']}."

    # Kiểm tra mặt nạ nhị phân
    mask = res["binary_mask"]
    assert mask.shape == synthetic_leaf_gray.shape, "Mặt nạ nhị phân phải khớp kích thước ảnh gốc."
    assert mask.dtype == np.uint8, "Mặt nạ nhị phân phải có kiểu np.uint8."
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255}), f"Mặt nạ nhị phân chỉ được chứa giá trị {{0, 255}}, thực tế: {unique_vals}."

    # Kiểm tra đường cong phương sai
    var_curve = res["variance_curve"]
    assert var_curve.shape == (256,), "Đường cong phương sai phải có kích thước (256,)."
    assert res["exec_time_ms"] > 0, "Thời gian thực thi phải lớn hơn 0."


# =============================================================================
# 4. TEST K-MEANS CONTRACTS
# =============================================================================

def test_kmeans_contract(synthetic_leaf_rgb: np.ndarray, synthetic_leaf_mask: np.ndarray):
    """Kiểm tra hợp đồng thuật toán gom cụm K-means vector hóa."""
    k = 3
    try:
        res = kmeans_scratch(synthetic_leaf_rgb, k=k, max_iter=30, roi_mask=synthetic_leaf_mask)
    except NotImplementedError:
        pytest.skip("kmeans_scratch chưa được triển khai.")

    required_keys = {"segmented_img", "labels_2d", "centroids", "n_iter", "exec_time_ms"}
    assert required_keys.issubset(res.keys()), f"Thiếu khóa trong kết quả K-means: {required_keys - set(res.keys())}"

    # segmented_img
    seg = res["segmented_img"]
    assert seg.shape == synthetic_leaf_rgb.shape, "Ảnh phân đoạn K-means phải có cùng kích thước (H, W, 3)."
    assert seg.dtype == np.uint8, "Ảnh phân đoạn phải có kiểu np.uint8."

    # labels_2d
    labels = res["labels_2d"]
    assert labels.shape == synthetic_leaf_rgb.shape[:2], "Bản đồ nhãn phải có kích thước (H, W)."
    assert labels.dtype == np.int32, "Bản đồ nhãn phải có kiểu np.int32."
    unique_labels = set(np.unique(labels))
    allowed_labels = set(range(-1, k))
    assert unique_labels.issubset(allowed_labels), f"Nhãn ngoài dải cho phép [-1..k-1]: {unique_labels}."

    # centroids
    centroids = res["centroids"]
    assert centroids.shape == (k, 3), f"Tọa độ tâm cụm phải là (k, 3), thực tế: {centroids.shape}."
    assert centroids.dtype == np.float32, "Tâm cụm phải có kiểu np.float32."

    # n_iter & exec_time_ms
    assert 1 <= res["n_iter"] <= 30, "Số vòng lặp phải nằm trong đoạn [1, max_iter]."
    assert res["exec_time_ms"] > 0, "Thời gian thực thi phải là số dương."


# =============================================================================
# 5. TEST COLOR CLUSTERING CONTRACTS
# =============================================================================

def test_color_clustering_contract(synthetic_leaf_rgb: np.ndarray, synthetic_leaf_mask: np.ndarray):
    """Kiểm tra hợp đồng phân tích màu tách cụm bệnh và mô khỏe."""
    k = 3
    labels_2d = np.zeros(synthetic_leaf_rgb.shape[:2], dtype=np.int32)
    # Gán nhãn giả định
    labels_2d[synthetic_leaf_mask > 0] = 0
    centroids = np.array([[34.0, 139.0, 34.0], [139.0, 69.0, 19.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    try:
        res = extract_disease_and_healthy_masks(
            synthetic_leaf_rgb, labels_2d, centroids, leaf_mask=synthetic_leaf_mask
        )
    except NotImplementedError:
        pytest.skip("extract_disease_and_healthy_masks chưa được triển khai.")

    required_keys = {"disease_mask", "healthy_mask", "disease_cluster_ids"}
    assert required_keys.issubset(res.keys()), f"Thiếu khóa trong kết quả tách màu: {required_keys - set(res.keys())}"

    for mask_key in ["disease_mask", "healthy_mask"]:
        mask = res[mask_key]
        assert mask.shape == synthetic_leaf_rgb.shape[:2], f"{mask_key} phải có kích thước (H, W)."
        assert mask.dtype == np.uint8, f"{mask_key} phải có kiểu np.uint8."
        assert set(np.unique(mask)).issubset({0, 255}), f"{mask_key} chỉ được chứa giá trị {{0, 255}}."

    assert isinstance(res["disease_cluster_ids"], list), "disease_cluster_ids phải là danh sách list."


# =============================================================================
# 6. TEST POSTPROCESS CONTRACTS
# =============================================================================

def test_morphology_contract():
    """Kiểm tra hợp đồng các hàm hình thái học nhị phân (Erode, Dilate, Opening, Closing)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    for morph_func in [erode_scratch, dilate_scratch, opening_scratch, closing_scratch]:
        try:
            out = morph_func(mask, kernel_size=3)
            assert out.shape == mask.shape, f"{morph_func.__name__} phải bảo toàn kích thước."
            assert out.dtype == np.uint8, f"{morph_func.__name__} phải trả về kiểu np.uint8."
            assert set(np.unique(out)).issubset({0, 255}), f"{morph_func.__name__} chỉ chứa {{0, 255}}."
        except NotImplementedError:
            pytest.skip(f"{morph_func.__name__} chưa được triển khai.")


def test_connected_components_bfs_contract():
    """Kiểm tra hợp đồng thuật toán BFS tìm thành phần liên thông đếm ổ bệnh."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Tạo 2 ổ bệnh hợp lệ (diện tích > 20)
    mask[10:20, 10:20] = 255  # Diện tích 100 px
    mask[40:50, 40:50] = 255  # Diện tích 100 px
    # Tạo 1 đốm nhiễu nhỏ diện tích 4 px
    mask[80:82, 80:82] = 255

    try:
        res = connected_components_bfs_scratch(mask, min_area=20)
    except NotImplementedError:
        pytest.skip("connected_components_bfs_scratch chưa được triển khai.")

    required_keys = {"cleaned_mask", "num_lesions", "bboxes"}
    assert required_keys.issubset(res.keys()), f"Thiếu khóa trong kết quả BFS: {required_keys - set(res.keys())}"

    # cleaned_mask
    cmask = res["cleaned_mask"]
    assert cmask.shape == mask.shape, "Cleaned mask phải có cùng kích thước (H, W)."
    assert cmask.dtype == np.uint8, "Cleaned mask phải có kiểu np.uint8."
    assert set(np.unique(cmask)).issubset({0, 255}), "Cleaned mask chỉ chứa {0, 255}."

    # num_lesions và bboxes
    assert isinstance(res["num_lesions"], int), "Số lượng ổ bệnh num_lesions phải là int."
    assert len(res["bboxes"]) == res["num_lesions"], "Số lượng BBox phải bằng num_lesions."

    for bbox in res["bboxes"]:
        assert len(bbox) == 5, f"BBox phải có đúng 5 thành phần (x, y, w, h, area), thực tế: {bbox}."
        x, y, w, h, area = bbox
        assert area >= 20, f"Diện tích ổ bệnh phải >= min_area (20), thực tế: {area}."
        assert w > 0 and h > 0, "Kích thước w, h của BBox phải dương."


# =============================================================================
# 7. TEST BENCHMARK CONTRACTS
# =============================================================================

def test_benchmark_metrics_contract():
    """Kiểm tra hợp đồng tính chỉ số phân đoạn IoU, Dice, Precision, Recall, Accuracy."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:60, 20:60] = 255
    mask2[30:70, 30:70] = 255

    try:
        metrics = compute_segmentation_metrics_scratch(mask1, mask2)
    except NotImplementedError:
        pytest.skip("compute_segmentation_metrics_scratch chưa được triển khai.")

    required_keys = {"iou", "dice", "precision", "recall", "accuracy"}
    assert required_keys.issubset(metrics.keys()), f"Thiếu chỉ số: {required_keys - set(metrics.keys())}"

    for key, val in metrics.items():
        assert isinstance(val, (float, np.floating)), f"Chỉ số {key} phải là float."
        assert 0.0 <= val <= 1.0, f"Chỉ số {key} phải nằm trong khoảng [0.0, 1.0], thực tế: {val}."


def test_diagnosis_contract(synthetic_leaf_mask: np.ndarray):
    """Kiểm tra hợp đồng hàm chẩn đoán 5 cấp độ bệnh lá cây."""
    disease_mask = np.zeros_like(synthetic_leaf_mask)
    # Tạo vết bệnh chiếm khoảng 10% diện tích phiến lá (Cấp 2)
    disease_mask[55:65, 55:65] = 255

    try:
        diag = diagnose_leaf_severity(synthetic_leaf_mask, disease_mask, num_lesions=1)
    except NotImplementedError:
        pytest.skip("diagnose_leaf_severity chưa được triển khai.")

    required_keys = {
        "severity_level",
        "severity_label",
        "disease_ratio_percent",
        "num_lesions",
        "leaf_area_px",
        "disease_area_px",
        "recommendation",
    }
    assert required_keys.issubset(diag.keys()), f"Thiếu khóa trong kết quả chẩn đoán: {required_keys - set(diag.keys())}"

    assert 0 <= diag["severity_level"] <= 4, f"Cấp độ bệnh phải từ 0 đến 4, thực tế: {diag['severity_level']}."
    assert 0.0 <= diag["disease_ratio_percent"] <= 100.0, "Tỷ lệ bệnh phải nằm trong [0.0, 100.0]."
    assert isinstance(diag["recommendation"], str), "Khuyến nghị điều trị phải là dạng chuỗi str."
    assert len(diag["recommendation"]) > 0, "Khuyến nghị không được để trống."


# =============================================================================
# 8. TEST INTEGRATION PIPELINE CONTRACT
# =============================================================================

def test_pipeline_integration_contract(synthetic_leaf_rgb: np.ndarray):
    """Kiểm tra hợp đồng hàm điều phối toàn trình pipeline.py."""
    try:
        result = run_full_diagnosis_pipeline(
            file_source=synthetic_leaf_rgb,
            k_clusters=3,
            blur_kernel=5,
            min_lesion_area=20,
        )
    except NotImplementedError:
        pytest.skip("run_full_diagnosis_pipeline chưa được tích hợp hoàn chỉnh.")

    required_stages = {
        "original_rgb",
        "grayscale",
        "otsu_result",
        "kmeans_result",
        "color_result",
        "postprocess_result",
        "diagnosis",
    }
    assert required_stages.issubset(result.keys()), f"Thiếu kết quả các tầng trong pipeline: {required_stages - set(result.keys())}"
