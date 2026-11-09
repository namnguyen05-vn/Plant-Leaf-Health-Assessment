"""Data Input/Output Module.

Module phụ trách việc đọc/ghi ảnh thực tế (.png, .jpg) và chuyển đổi hai chiều
giữa mảng số NumPy và file ma trận (.csv) phục vụ trực quan hóa dữ liệu điểm ảnh.
"""

from pathlib import Path
from typing import Any, Union
import numpy as np


def load_image_or_csv(
    file_source: Union[str, Path, Any],
    max_size: int = 512,
) -> np.ndarray:
    """Đọc ảnh từ đường dẫn file hoặc đối tượng upload và chuẩn hóa về mảng NumPy uint8.

    Hỗ trợ đọc các định dạng ảnh tiêu chuẩn (.jpg, .jpeg, .png) và file ma trận
    điểm ảnh (.csv). Tự động co dãn (downscale) bảo toàn tỷ lệ khung hình nếu
    kích thước chiều lớn nhất vượt quá `max_size`.

    Args:
        file_source: Đường dẫn tới file ảnh/CSV (str hoặc Path), hoặc đối tượng buffer
            nhị phân (ví dụ: `UploadedFile` từ Streamlit hoặc `io.BytesIO`).
        max_size: Kích thước tối đa của chiều dài hoặc chiều rộng sau khi đọc. Mặc định là 512.
            Nếu max(H, W) > max_size, ảnh sẽ được resize bảo toàn tỷ lệ (aspect ratio).

    Returns:
        np.ndarray:
            - Nếu là ảnh thông thường: Mảng 3 chiều không gian màu RGB, kích thước `(H, W, 3)`,
              kiểu dữ liệu `np.uint8`, giá trị trong đoạn `[0, 255]`.
            - Nếu là ma trận xám từ CSV: Mảng 2 chiều kích thước `(H, W)`, kiểu `np.uint8`.

    Raises:
        ValueError: Nếu định dạng file không được hỗ trợ hoặc dữ liệu ma trận không hợp lệ.
        FileNotFoundError: Nếu file_source là đường dẫn không tồn tại trên hệ thống.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.

    Note:
        - Đối với ảnh màu, bắt buộc trả về thứ tự kênh màu RGB (không dùng thứ tự BGR mặc định của OpenCV).
        - Khi đọc từ CSV, cần kiểm tra và ép kiểu giá trị về `np.uint8` [0..255].
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Xác định kiểu đầu vào của `file_source` (đường dẫn chuỗi / Path hay buffer).
    # Bước 2: Phân biệt đuôi mở rộng (.csv vs .jpg/.png):
    #         - Nếu là CSV: Dùng pandas.read_csv hoặc numpy.loadtxt để parse ma trận số,
    #           sau đó ép kiểu sang np.uint8 và giới hạn [0, 255].
    #         - Nếu là ảnh: Dùng PIL.Image.open hoặc cv2.imdecode / cv2.imread.
    #           LƯU Ý: Nếu dùng OpenCV phải chuyển cv2.COLOR_BGR2RGB.
    # Bước 3: Kiểm tra kích thước (H, W). Nếu max(H, W) > max_size:
    #         Tính scale factor và resize theo phép nội suy song tuyến (Bilinear).
    # Bước 4: Trả về mảng numpy.ndarray với dtype=np.uint8.
    # =========================================================================
    raise NotImplementedError("Hàm load_image_or_csv cần được cài đặt trong nhánh feat/data-io.")


def save_matrix_to_csv(
    matrix_2d: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Lưu ma trận 2D số nguyên (mức xám, nhãn phân cụm, hoặc mặt nạ) ra file CSV.

    Args:
        matrix_2d: Mảng 2 chiều kích thước `(H, W)`, kiểu số nguyên (`uint8` hoặc `int32`),
            đại diện cho giá trị mức xám, nhãn cụm K-means hoặc binary mask.
        output_path: Đường dẫn file đích lưu trữ (.csv).

    Raises:
        ValueError: Nếu `matrix_2d` không phải mảng 2 chiều (ndim != 2).
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Kiểm tra `matrix_2d.ndim == 2`.
    # Bước 2: Tạo thư mục cha nếu chưa tồn tại (Path(output_path).parent.mkdir(parents=True, exist_ok=True)).
    # Bước 3: Dùng np.savetxt(output_path, matrix_2d, fmt='%d', delimiter=',') hoặc pandas DataFrame.
    # =========================================================================
    raise NotImplementedError("Hàm save_matrix_to_csv cần được cài đặt trong nhánh feat/data-io.")


def matrix_to_csv_bytes(matrix_2d: np.ndarray) -> bytes:
    """Chuyển đổi ma trận 2D thành chuỗi bytes định dạng CSV để phục vụ tải xuống trên Streamlit.

    Args:
        matrix_2d: Mảng 2 chiều kích thước `(H, W)` với kiểu dữ liệu số nguyên.

    Returns:
        bytes: Dữ liệu file CSV được mã hóa UTF-8 bytes.

    Raises:
        ValueError: Nếu `matrix_2d` không phải mảng 2 chiều.
        NotImplementedError: Khi hàm chưa được hiện thực hoàn chỉnh trên nhánh tính năng.
    """
    # =========================================================================
    # HƯỚNG DẪN TRIỂN KHAI CHO THÀNH VIÊN:
    # -------------------------------------------------------------------------
    # Bước 1: Kiểm tra ma trận 2D.
    # Bước 2: Sử dụng io.StringIO hoặc io.BytesIO kết hợp np.savetxt(..., fmt='%d', delimiter=',').
    # Bước 3: Trả về chuỗi bytes sau khi encode('utf-8').
    # =========================================================================
    raise NotImplementedError("Hàm matrix_to_csv_bytes cần được cài đặt trong nhánh feat/data-io.")
