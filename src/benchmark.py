"""Quantitative Benchmarking, Evaluation Metrics, and Clinical Diagnosis Module.

Module phụ trách việc đánh giá định lượng thuật toán và chẩn đoán sức khỏe lá cây:
1. Tính các chỉ số đánh giá phân đoạn ảnh nhị phân: TP, Union, IoU, Dice Coefficient (F1-score),
   Pixel Accuracy, Precision, Recall, MSE, PSNR hoàn toàn bằng NumPy from scratch.
2. Đo độ tương đồng ảnh: MSE (Mean Squared Error), PSNR (Peak Signal-to-Noise Ratio).
3. Triển khai các hàm quy chiếu đối sánh với OpenCV chuẩn (`cv2.threshold`, `cv2.kmeans`)
   để so sánh độ chính xác và tốc độ thực thi giữa thuật toán tự viết (scratch) và thư viện C++.
4. Phân loại 5 cấp độ bệnh lá cây dựa trên tỷ lệ phần trăm diện tích nhiễm bệnh và đưa ra khuyến nghị xử lý.
"""

import time
from typing import Any, Dict, Optional
import numpy as np
import cv2


def compute_segmentation_metrics_scratch(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> Dict[str, float]:
    """Tính các chỉ số định lượng đánh giá phân đoạn ảnh nhị phân từ đầu bằng NumPy.

    Công thức toán học:
        - TP (Intersection): Số pixel đúng là tiền cảnh (pred > 0 and ref > 0)
        - FP: Số pixel dự đoán nhầm thành tiền cảnh (pred > 0 and ref == 0)
        - FN: Số pixel bỏ sót tiền cảnh (pred == 0 and ref > 0)
        - TN: Số pixel đúng là hậu cảnh (pred == 0 and ref == 0)
        - Union = TP + FP + FN
        - IoU (Jaccard) = TP / Union
        - Dice (F1-score) = 2 * TP / (2 * TP + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - MSE (thang [0, 255]) và PSNR (dB, trả về 100.0 nếu MSE == 0)

    Args:
        pred_mask: Mặt nạ phân đoạn dự đoán của thuật toán, kích thước `(H, W)`, kiểu `np.uint8` `{0, 255}`.
        ref_mask: Mặt nạ tham chiếu chuẩn (Ground Truth hoặc OpenCV baseline), kích thước `(H, W)`, kiểu `np.uint8` `{0, 255}`.

    Returns:
        Dict[str, float]: Dictionary chứa cả chỉ số chuẩn hóa `[0.0, 1.0]` và dạng phần trăm `%`:
            - "iou": Jaccard Index trong `[0.0, 1.0]`.
            - "dice": Dice Score trong `[0.0, 1.0]`.
            - "precision": Precision trong `[0.0, 1.0]`.
            - "recall": Recall trong `[0.0, 1.0]`.
            - "accuracy": Pixel Accuracy trong `[0.0, 1.0]`.
            - "iou_pct", "dice_pct", "pixel_accuracy_pct", "precision_pct", "recall_pct": Thang `[0.0, 100.0]`.
            - "tp", "union": Số pixel giao (TP) và hợp (Union).
            - "mse", "psnr": Sai số toàn phương trung bình và PSNR (dB).

    Raises:
        ValueError: Nếu kích thước 2 mặt nạ không khớp nhau.
    """
    if pred_mask.shape[:2] != ref_mask.shape[:2]:
        raise ValueError(
            f"Kích thước pred_mask {pred_mask.shape} không khớp với ref_mask {ref_mask.shape}."
        )

    p = pred_mask > 0
    r = ref_mask > 0

    tp = int(np.sum(p & r))
    fp = int(np.sum(p & (~r)))
    fn = int(np.sum((~p) & r))
    tn = int(np.sum((~p) & (~r)))

    union = tp + fp + fn
    total = tp + fp + fn + tn

    iou = 1.0 if union == 0 else float(tp / union)
    dice_denom = 2 * tp + fp + fn
    dice = 1.0 if dice_denom == 0 else float((2.0 * tp) / dice_denom)

    if tp + fp == 0:
        precision = 1.0 if fn == 0 else 0.0
    else:
        precision = float(tp / (tp + fp))

    if tp + fn == 0:
        recall = 1.0 if fp == 0 else 0.0
    else:
        recall = float(tp / (tp + fn))

    accuracy = float((tp + tn) / max(total, 1))

    # Tính MSE và PSNR trên thang nhị phân [0, 255]
    p_val = np.where(p, 255.0, 0.0)
    r_val = np.where(r, 255.0, 0.0)
    mse = float(np.mean((p_val - r_val) ** 2))
    psnr = 100.0 if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "iou_pct": round(iou * 100.0, 2),
        "dice_pct": round(dice * 100.0, 2),
        "pixel_accuracy_pct": round(accuracy * 100.0, 2),
        "precision_pct": round(precision * 100.0, 2),
        "recall_pct": round(recall * 100.0, 2),
        "tp": float(tp),
        "union": float(union),
        "mse": round(mse, 4),
        "psnr": round(psnr, 4),
    }


def compute_image_similarity_metrics(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Dict[str, float]:
    """Tính các chỉ số tương đồng sai số toàn phương trung bình (MSE) và PSNR từ đầu bằng NumPy.

    Công thức:
        MSE = (1 / N) * sum((img1 - img2)^2)
        PSNR = 20 * log10(255.0 / sqrt(MSE))  (trả về 100.0 nếu MSE == 0)

    Args:
        img1: Mảng ảnh thứ nhất, kích thước `(H, W)` hoặc `(H, W, C)`, kiểu `np.uint8`.
        img2: Mảng ảnh thứ hai cùng kích thước, kiểu `np.uint8`.

    Returns:
        Dict[str, float]: Chứa các khóa `"mse"` và `"psnr"`.

    Raises:
        ValueError: Nếu kích thước hai ảnh không khớp nhau.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Kích thước ảnh không khớp: {img1.shape} vs {img2.shape}.")

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = 100.0 if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    return {
        "mse": round(mse, 4),
        "psnr": round(psnr, 4),
    }


def run_opencv_otsu(
    gray_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Chạy thuật toán Otsu baseline của OpenCV để đối sánh thời gian và kết quả phân đoạn.

    Args:
        gray_img: Mảng ảnh mức xám 2 chiều kích thước `(H, W)`, kiểu `np.uint8`.
        roi_mask: (Tùy chọn) Vùng quan tâm ROI `(H, W)`.

    Returns:
        Dict[str, Any]: Chứa các khóa:
            - "threshold": int, ngưỡng Otsu do OpenCV tính.
            - "binary_mask": np.ndarray `(H, W)`, kiểu `np.uint8` `{0, 255}`.
            - "exec_time_ms": float, thời gian thực thi bằng ms.

    Raises:
        ValueError: Nếu `gray_img` không phải mảng 2 chiều hoặc `roi_mask` không khớp kích thước.
    """
    if gray_img.ndim != 2:
        raise ValueError(f"gray_img phải là mảng 2 chiều (H, W), nhận được: {gray_img.shape}.")

    gray_u8 = gray_img.astype(np.uint8, copy=False)
    t_start = time.perf_counter()

    if roi_mask is not None:
        if roi_mask.shape[:2] != gray_u8.shape:
            raise ValueError(f"roi_mask {roi_mask.shape} không khớp gray_img {gray_u8.shape}.")
        valid = roi_mask > 0
        roi_pixels = gray_u8[valid]
        if roi_pixels.size == 0:
            thresh_val = 0
            binary_mask = np.zeros_like(gray_u8, dtype=np.uint8)
        else:
            thresh_f, _ = cv2.threshold(
                roi_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresh_val = int(round(thresh_f))
            binary_mask = np.zeros_like(gray_u8, dtype=np.uint8)
            binary_mask[valid & (gray_u8 > thresh_val)] = 255
    else:
        thresh_f, binary_mask = cv2.threshold(
            gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh_val = int(round(thresh_f))

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "threshold": thresh_val,
        "binary_mask": binary_mask.astype(np.uint8),
        "exec_time_ms": float(exec_time_ms),
    }


def run_opencv_kmeans(
    img: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-3,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Chạy thuật toán K-means baseline tối ưu hóa bằng OpenCV C++ để đối sánh hiệu năng.

    Args:
        img: Mảng ảnh đầu vào `(H, W, C)` hoặc `(H, W)`, kiểu `np.uint8`.
        k: Số cụm K (>= 2).
        max_iter: Số vòng lặp tối đa.
        tol: Dung sai hội tụ.
        roi_mask: (Tùy chọn) Mặt nạ ROI `(H, W)`.

    Returns:
        Dict[str, Any]: Chứa các khóa:
            - "segmented_img": np.ndarray cùng kích thước `img`, kiểu `np.uint8`.
            - "labels_2d": np.ndarray `(H, W)`, kiểu `np.int32`.
            - "centroids": np.ndarray `(k, C)`, kiểu `np.float32`.
            - "exec_time_ms": float.

    Raises:
        ValueError: Nếu `k < 2` hoặc kích thước đầu vào không hợp lệ.
    """
    if k < 2:
        raise ValueError(f"Số cụm k phải >= 2, nhận được: {k}.")
    if img.ndim not in (2, 3):
        raise ValueError(f"Ảnh đầu vào phải là 2D hoặc 3D, nhận được: {img.shape}.")

    H, W = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]

    t_start = time.perf_counter()

    if roi_mask is not None:
        if roi_mask.shape[:2] != (H, W):
            raise ValueError(f"roi_mask {roi_mask.shape} không khớp ảnh {(H, W)}.")
        valid_mask = roi_mask > 0
        pixels = img[valid_mask].reshape(-1, C).astype(np.float32)
    else:
        valid_mask = None
        pixels = img.reshape(-1, C).astype(np.float32)

    if pixels.shape[0] == 0:
        exec_time_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "segmented_img": np.zeros_like(img, dtype=np.uint8),
            "labels_2d": np.full((H, W), -1, dtype=np.int32),
            "centroids": np.zeros((k, C), dtype=np.float32),
            "exec_time_ms": float(exec_time_ms),
        }

    k_eff = min(k, pixels.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), float(tol))
    _, labels, centers = cv2.kmeans(
        pixels, k_eff, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    labels_flat = labels.ravel().astype(np.int32)
    centroids = np.zeros((k, C), dtype=np.float32)
    centroids[:k_eff] = centers.astype(np.float32)

    centers_u8 = np.clip(np.round(centroids), 0, 255).astype(np.uint8)

    if valid_mask is not None:
        labels_2d = np.full((H, W), -1, dtype=np.int32)
        labels_2d[valid_mask] = labels_flat
        if img.ndim == 2:
            segmented_img = np.zeros((H, W), dtype=np.uint8)
            segmented_img[valid_mask] = centers_u8[labels_flat, 0]
        else:
            segmented_img = np.zeros((H, W, C), dtype=np.uint8)
            segmented_img[valid_mask] = centers_u8[labels_flat]
    else:
        labels_2d = labels_flat.reshape(H, W)
        if img.ndim == 2:
            segmented_img = centers_u8[labels_flat, 0].reshape(H, W)
        else:
            segmented_img = centers_u8[labels_flat].reshape(H, W, C)

    exec_time_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "segmented_img": segmented_img,
        "labels_2d": labels_2d,
        "centroids": centroids,
        "exec_time_ms": float(exec_time_ms),
    }


def diagnose_leaf_severity(
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    num_lesions: int,
) -> Dict[str, Any]:
    """Chẩn đoán mức độ nhiễm bệnh của lá cây theo thang phân loại 5 cấp độ chuẩn nông nghiệp.

    Công thức tính tỷ lệ diện tích tổn thương:
        infection_ratio_pct = (disease_area_px / max(leaf_area_px, 1)) * 100.0

    Thang 5 cấp độ chẩn đoán:
    - Cấp 0 — Lá khỏe mạnh (`infection_ratio_pct < 1.0%`)
    - Cấp 1 — Nhiễm bệnh nhẹ (`1.0% <= infection_ratio_pct < 10.0%`)
    - Cấp 2 — Nhiễm bệnh trung bình (`10.0% <= infection_ratio_pct < 25.0%`)
    - Cấp 3 — Nhiễm bệnh nặng (`25.0% <= infection_ratio_pct < 50.0%`)
    - Cấp 4 — Nhiễm bệnh nghiêm trọng (`infection_ratio_pct >= 50.0%`)

    Args:
        leaf_mask: Mặt nạ diện tích toàn bộ phiến lá `(H, W)`, kiểu `np.uint8` `{0, 255}`.
        disease_mask: Mặt nạ vùng vết bệnh `(H, W)`, kiểu `np.uint8` `{0, 255}`.
        num_lesions: Số lượng ổ bệnh độc lập đã phát hiện (từ thuật toán BFS).

    Returns:
        Dict[str, Any]: Dictionary chuẩn gồm các thông tin chẩn đoán y bạ cây trồng:
            - "severity_level": int (0 đến 4).
            - "severity_label": str (mô tả cấp độ).
            - "disease_ratio_percent": float, tỷ lệ % diện tích bệnh `[0.0, 100.0]`.
            - "infection_ratio_pct": float, bí danh đồng nhất với `disease_ratio_percent`.
            - "num_lesions": int, số lượng ổ bệnh độc lập.
            - "leaf_area_px": int, tổng số pixel của phiến lá.
            - "disease_area_px": int, tổng số pixel bị nhiễm bệnh.
            - "recommendation": str, phác đồ điều trị và khuyến nghị nông nghiệp cụ thể.

    Raises:
        ValueError: Nếu kích thước `leaf_mask` và `disease_mask` không khớp nhau.
    """
    if leaf_mask.shape[:2] != disease_mask.shape[:2]:
        raise ValueError(
            f"Kích thước leaf_mask {leaf_mask.shape} không khớp disease_mask {disease_mask.shape}."
        )

    leaf_bool = leaf_mask > 0
    disease_bool = (disease_mask > 0) & leaf_bool

    leaf_area_px = int(np.sum(leaf_bool))
    disease_area_px = int(np.sum(disease_bool))
    infection_ratio_pct = float((disease_area_px / max(leaf_area_px, 1)) * 100.0)
    infection_ratio_pct = min(100.0, max(0.0, infection_ratio_pct))

    if infection_ratio_pct < 1.0:
        level = 0
        label = "Cấp 0 - Lá khỏe mạnh"
        recommendation = (
            "Lá cây ở trạng thái khỏe mạnh, không có tổn thương bệnh lý đáng kể (< 1.0%). "
            "Tiếp tục duy trì chế độ dinh dưỡng, ánh sáng và tưới tiêu hiện tại."
        )
    elif infection_ratio_pct < 10.0:
        level = 1
        label = "Cấp 1 - Nhiễm bệnh nhẹ"
        recommendation = (
            "Bệnh mới khởi phát ở mức độ nhẹ (1.0% - < 10.0%). "
            "Khuyến nghị tỉa bỏ các đốm/lá bị nhiễm sớm, tăng cường độ thông thoáng tán lá và theo dõi định kỳ."
        )
    elif infection_ratio_pct < 25.0:
        level = 2
        label = "Cấp 2 - Nhiễm bệnh trung bình"
        recommendation = (
            "Vết bệnh đang lây lan ở mức trung bình (10.0% - < 25.0%). "
            "Khuyến nghị phun chế phẩm sinh học hoặc thuốc bảo vệ thực vật phù hợp và tránh tưới nước trực tiếp lên tán lá."
        )
    elif infection_ratio_pct < 50.0:
        level = 3
        label = "Cấp 3 - Nhiễm bệnh nặng"
        recommendation = (
            "Diện tích mô lá tổn thương lớn (25.0% - < 50.0%). "
            "Cần cách ly khu vực nhiễm bệnh, cắt tỉa lá nặng và sử dụng thuốc đặc trị theo hướng dẫn kỹ thuật."
        )
    else:
        level = 4
        label = "Cấp 4 - Nhiễm bệnh nghiêm trọng"
        recommendation = (
            "Phiến lá bị hoại tử nghiêm trọng (>= 50.0%), suy giảm nghiêm trọng khả năng quang hợp. "
            "Cần thu gom và tiêu hủy ngay để phòng tránh lây lan diện rộng."
        )

    ratio_rounded = round(infection_ratio_pct, 2)

    return {
        "severity_level": level,
        "severity_label": label,
        "disease_ratio_percent": ratio_rounded,
        "infection_ratio_pct": ratio_rounded,
        "num_lesions": int(num_lesions),
        "leaf_area_px": leaf_area_px,
        "disease_area_px": disease_area_px,
        "recommendation": recommendation,
    }
