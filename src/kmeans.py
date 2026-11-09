"""K-Means Clustering Module (Vectorized from Scratch with NumPy).

Module triển khai thuật toán phân cụm K-means từ đầu bằng kỹ thuật vector hóa NumPy:
1. Khởi tạo tâm cụm (Centroids) ngẫu nhiên có kiểm soát hạt giống (random_state) hoặc theo K-means++.
2. Tính toán ma trận khoảng cách Euclid đa chiều vector hóa giữa tất cả điểm ảnh và các tâm cụm.
3. Gán nhãn cụm (Cluster Assignment) và tái tính toán vị trí tâm cụm mới (Centroid Update).
4. Kiểm tra điều kiện hội tụ theo ngưỡng dung sai sai số (tolerance) hoặc số vòng lặp tối đa.
5. Hỗ trợ phân cụm giới hạn trong vùng mặt nạ quan tâm (ROI Mask) để loại bỏ nền trước khi gom cụm.

QUY TẮC BẮT BUỘC (CODE FROM SCRATCH):
- TUYỆT ĐỐI KHÔNG sử dụng `cv2.kmeans` hay `sklearn.cluster.KMeans`.
- Bắt buộc dùng NumPy broadcasting để vector hóa khoảng cách, tránh lặp qua từng điểm ảnh.
"""

import time
from typing import Any, Dict, Optional
import numpy as np


def initialize_centroids(
    features: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Khởi tạo tọa độ K tâm cụm ban đầu từ tập vector đặc trưng.

    Args:
        features: Mảng 2 chiều chứa N điểm dữ liệu, kích thước `(N, C)`, kiểu `float32`.
        k: Số lượng cụm cần phân chia (k >= 2).
        random_state: Hạt giống ngẫu nhiên phục vụ tái lập kết quả thử nghiệm.

    Returns:
        np.ndarray: Mảng 2 chiều chứa tọa độ K tâm cụm khởi tạo, kích thước `(k, C)`, kiểu `float32`.

    Raises:
        ValueError: Nếu số điểm dữ liệu N nhỏ hơn k.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Khởi tạo bộ sinh ngẫu nhiên np.random.RandomState(random_state).
    # Bước 2: Chọn ngẫu nhiên không lặp lại k chỉ số từ 0 đến N-1: indices = rng.choice(N, size=k, replace=False).
    # Bước 3: Lấy các vector đặc trưng tương ứng làm tâm cụm ban đầu: centroids = features[indices].copy().
    # =========================================================================
    raise NotImplementedError("Hàm initialize_centroids cần được cài đặt trong nhánh feat/kmeans.")


def compute_euclidean_distances(
    features: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Tính toán ma trận khoảng cách Euclid vector hóa giữa N điểm dữ liệu và K tâm cụm.

    Áp dụng kỹ thuật NumPy broadcasting:
        d_ij = ||x_i - c_j||_2 = sqrt(sum_c (x_ic - c_jc)^2)

    Args:
        features: Mảng 2 chiều kích thước `(N, C)`, kiểu `float32`.
        centroids: Mảng 2 chiều kích thước `(k, C)`, kiểu `float32`.

    Returns:
        np.ndarray: Ma trận khoảng cách 2 chiều kích thước `(N, k)`, kiểu `float32`,
            trong đó phần tử tại hàng i, cột j là khoảng cách từ điểm i đến tâm cụm j.

    Raises:
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Dùng broadcasting `features[:, np.newaxis, :] - centroids[np.newaxis, :, :]`.
        - Không sử dụng vòng lặp Python duyệt qua N phần tử để đảm bảo thời gian chạy dưới 1-2 giây.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Mở rộng trục: diff = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # Shape: (N, k, C)
    # Bước 2: Tính bình phương tổng: dist_sq = np.sum(diff ** 2, axis=2)  # Shape: (N, k)
    # Bước 3: Trả về căn bậc hai: np.sqrt(dist_sq) (hoặc tối ưu giữ dist_sq để tìm argmin).
    # =========================================================================
    raise NotImplementedError("Hàm compute_euclidean_distances cần được cài đặt trong nhánh feat/kmeans.")


def kmeans_scratch(
    img: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Thuật toán phân cụm K-means từ đầu bằng NumPy vector hóa để phân đoạn màu lá cây.

    Quy trình phân đoạn:
    1. Nếu có `roi_mask`, chỉ lấy tập các pixel thuộc tiền cảnh lá (`roi_mask > 0`) để gom cụm.
    2. Chuyển đổi ma trận ảnh sang mảng 2 chiều đặc trưng màu `(N, C)` kiểu `float32`.
    3. Khởi tạo K tâm cụm ban đầu.
    4. Lặp tối đa `max_iter` lần:
        - Gán nhãn mỗi điểm ảnh về tâm cụm gần nhất: `label = argmin_j ||x_i - c_j||^2`.
        - Cập nhật tọa độ tâm cụm: `c_j = mean(x_i thuộc cụm j)`.
        - Kiểm tra mức độ dịch chuyển của tâm cụm: nếu `max(||c_new - c_old||) < tol` thì dừng sớm.
    5. Tái tạo lại ảnh phân đoạn `segmented_img` và bản đồ nhãn 2D `labels_2d`.

    Args:
        img: Mảng ảnh đầu vào kích thước `(H, W, C)` hoặc `(H, W)`, kiểu `np.uint8`.
            Thường là không gian màu RGB hoặc kết hợp đặc trưng HSV/Lab.
        k: Số lượng cụm cần phân tách (thường k=3: nền, mô lá khỏe, vết bệnh).
        max_iter: Số vòng lặp tối đa cho phép. Mặc định là 100.
        tol: Ngưỡng dung sai độ dịch chuyển tâm cụm để xác định hội tụ. Mặc định 1e-3.
        random_state: Seed ngẫu nhiên phục vụ tái lập kết quả. Mặc định 42.
        roi_mask: (Tùy chọn) Mặt nạ 2 chiều `(H, W)` nhị phân `{0, 255}` hoặc boolean.
            Nếu cung cấp, chỉ gom cụm trong vùng `roi_mask > 0`. Các pixel ngoài ROI
            sẽ được gán nhãn `-1` và hiển thị màu đen `[0, 0, 0]` trên ảnh phân đoạn.

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa các khóa:
            - "segmented_img": np.ndarray cùng kích thước `(H, W, C)` với ảnh đầu vào,
              kiểu `np.uint8`, giá trị mỗi pixel được gán bằng màu tâm cụm tương ứng.
            - "labels_2d": np.ndarray kích thước `(H, W)`, kiểu `np.int32`, giá trị mỗi phần tử
              nằm trong `[0, k - 1]`, hoặc `-1` cho vùng bị loại trừ bởi `roi_mask`.
            - "centroids": np.ndarray kích thước `(k, C)`, kiểu `np.float32`, tọa độ các tâm cụm.
            - "n_iter": int, số vòng lặp thực tế đã thực hiện trước khi dừng.
            - "exec_time_ms": float, thời gian chạy thuật toán tính bằng mili-giây (ms).

    Raises:
        ValueError: Nếu k < 2 hoặc ảnh đầu vào không hợp lệ.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - QUY TẮC: Tuyệt đối không dùng `cv2.kmeans` hay thư viện `scipy` / `sklearn`.
        - Thuật toán phải hoàn tất dưới 3 giây cho ảnh 512x512 với k=3 trên CPU.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Ghi nhận thời gian bắt đầu t_start = time.perf_counter().
    # Bước 2: Chuẩn bị ma trận đặc trưng X:
    #         - Lọc pixel theo roi_mask (nếu có).
    #         - Chuyển thành ma trận 2D (N, C) kiểu float32.
    # Bước 3: Khởi tạo k tâm cụm centroids.
    # Bước 4: Vòng lặp tối đa max_iter:
    #         - distances = compute_euclidean_distances(X, centroids)
    #         - labels = np.argmin(distances, axis=1)
    #         - Cập nhật centroids mới bằng trung bình các điểm thuộc cụm.
    #         - Kiểm tra điều kiện dừng: np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol.
    # Bước 5: Chiếu nhãn 1D trở lại mảng 2D labels_2d (H, W). Điền -1 cho pixel ngoài roi_mask.
    # Bước 6: Tạo segmented_img bằng cách thay thế pixel bằng màu tâm cụm tương ứng.
    # Bước 7: Tính exec_time_ms và trả về kết quả theo Data Contract.
    # =========================================================================
    raise NotImplementedError("Hàm kmeans_scratch cần được cài đặt trong nhánh feat/kmeans.")
