"""Color Feature Analysis and Tissue Classification Module.

Module phân tích tọa độ không gian màu của các tâm cụm K-means để tự động phân định
cụm nào đại diện cho mô lá khỏe mạnh (Healthy Tissue) và cụm nào đại diện cho vết bệnh/hoại tử (Diseased Lesions).
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_excess_green_index(rgb: np.ndarray) -> np.ndarray:
    """Tính chỉ số sắc xanh vượt trội ExG (Excess Green Index) từ vector màu RGB.

    Công thức:
        ExG = 2 * G - R - B

    Mô lá khỏe mạnh có hàm lượng diệp lục cao nên ExG > 0 và đạt giá trị lớn.
    Các vết bệnh (nâu sẫm, vàng úa, đốm đen) có sắc tố lục suy giảm nên ExG thấp hoặc âm.

    Args:
        rgb: Mảng 1D hoặc 2D chứa các giá trị kênh R, G, B kiểu `float32` hoặc `uint8`.

    Returns:
        np.ndarray: Mảng giá trị ExG tương ứng, kiểu `float32`.

    Raises:
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Ép kiểu sang float32 để tránh tràn số.
    # Bước 2: R = rgb[..., 0], G = rgb[..., 1], B = rgb[..., 2].
    # Bước 3: exg = 2.0 * G - R - B. Trả về exg.
    # =========================================================================
    raise NotImplementedError("Hàm compute_excess_green_index cần được cài đặt trong nhánh feat/color-clustering.")


def extract_disease_and_healthy_masks(
    rgb_img: np.ndarray,
    labels_2d: np.ndarray,
    centroids: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Phân loại các cụm K-means thành vùng mô lá khỏe và vùng vết bệnh dựa trên phân tích màu.

    Quy tắc phân loại:
    1. Với mỗi tâm cụm c_j (j in [0..k-1]), tính toán các chỉ số sinh học:
       - Chỉ số xanh ExG_j = 2 * G_j - R_j - B_j.
       - Chỉ số nâu / vàng úa: Brown_j = R_j - B_j hoặc tỷ lệ R_j / (G_j + 1e-5).
    2. Cụm có ExG cao nhất được xếp vào lớp mô lá khỏe ("Healthy Leaf").
    3. Cụm có ExG thấp hơn và màu sắc lệch về gam màu nóng/tối (vàng, nâu, đen, đỏ)
       được xếp vào lớp vết bệnh ("Disease Lesion").
    4. Tổng hợp các pixel thuộc các cụm tương ứng để tạo ra hai mặt nạ nhị phân tách biệt.

    Args:
        rgb_img: Mảng ảnh gốc RGB 3 chiều kích thước `(H, W, 3)`, kiểu `np.uint8`.
        labels_2d: Bản đồ nhãn cụm 2 chiều kích thước `(H, W)`, kiểu `np.int32`
            với giá trị nhãn từ `0` đến `k-1` (hoặc `-1` nếu là điểm ngoài lá).
        centroids: Mảng 2 chiều kích thước `(k, C)` chứa màu tâm cụm, kiểu `np.float32`.
        leaf_mask: (Tùy chọn) Mặt nạ nhị phân lá cây `(H, W)`, kiểu `np.uint8` `{0, 255}`.
            Nếu cung cấp, chỉ xét phân loại trên vùng diện tích thuộc lá cây.

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa các khóa:
            - "disease_mask": np.ndarray kích thước `(H, W)`, kiểu `np.uint8`,
              chỉ nhận hai giá trị `{0, 255}`. Giá trị 255 đại diện cho các điểm ảnh vết bệnh.
            - "healthy_mask": np.ndarray kích thước `(H, W)`, kiểu `np.uint8`,
              chỉ nhận hai giá trị `{0, 255}`. Giá trị 255 đại diện cho mô lá xanh khỏe mạnh.
            - "disease_cluster_ids": List[int], danh sách chỉ số (ID) các cụm được xếp vào lớp bệnh.

    Raises:
        ValueError: Nếu kích thước `labels_2d` không khớp với `rgb_img`.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Xử lý bằng NumPy vector hóa, không dùng vòng lặp duyệt qua từng điểm ảnh.
        - Phải đảm bảo không nhận nhầm vùng nền (background) vào vùng bệnh bằng cách kiểm tra `leaf_mask`.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Tính chỉ số ExG cho từng centroid trong centroids.
    # Bước 2: Xác định cluster_healthy = int(np.argmax(exg_scores)).
    # Bước 3: Các cluster còn lại có ExG < ngưỡng hoặc tỷ lệ nâu/vàng cao được xếp vào disease_cluster_ids.
    # Bước 4: Tạo mặt nạ `healthy_mask`: khởi tạo mảng 0, gán 255 tại `labels_2d == cluster_healthy`.
    # Bước 5: Tạo mặt nạ `disease_mask`: gán 255 tại các pixel có label nằm trong disease_cluster_ids.
    # Bước 6: Nếu có leaf_mask, áp dụng phép toán AND bitwise:
    #         disease_mask = np.where(leaf_mask > 0, disease_mask, 0).astype(np.uint8)
    #         healthy_mask = np.where(leaf_mask > 0, healthy_mask, 0).astype(np.uint8)
    # Bước 7: Trả về kết quả theo chuẩn Data Contract.
    # =========================================================================
    raise NotImplementedError(
        "Hàm extract_disease_and_healthy_masks cần được cài đặt trong nhánh feat/color-clustering."
    )
