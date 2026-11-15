"""Data Input/Output Module.

Module phụ trách việc đọc/ghi ảnh thực tế (.png, .jpg) và chuyển đổi hai chiều
giữa mảng số NumPy và file ma trận (.csv) phục vụ trực quan hóa dữ liệu điểm ảnh.

Định dạng CSV của dự án:
- Mỗi dòng là một hàng điểm ảnh, các giá trị số nguyên ngăn cách bởi dấu phẩy, KHÔNG có header.
- Bộ đọc chấp nhận thêm dấu `;`, tab hoặc khoảng trắng làm dấu phân cách, bỏ qua dòng trống,
  dòng chú thích bắt đầu bằng `#` và một dòng header chữ (nếu có) ở đầu file.

Đọc ảnh dùng Pillow (được phép theo quy tắc dự án: OpenCV/PIL chỉ dùng cho I/O cơ bản).
Phần parse/ghi CSV được tự cài đặt, không phụ thuộc pandas.
"""

import io
import warnings
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_CSV_EXTS = {".csv", ".txt"}


# =============================================================================
# HÀM PHỤ TRỢ
# =============================================================================

def _read_source_bytes(file_source: Union[str, Path, Any]) -> tuple:
    """Trả về (bytes dữ liệu, tên file viết thường) từ đường dẫn / bytes / buffer."""
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_source}")
        if not path.is_file():
            raise ValueError(f"Đường dẫn không phải file: {file_source}")
        return path.read_bytes(), path.name.lower()

    if isinstance(file_source, (bytes, bytearray, memoryview)):
        return bytes(file_source), ""

    if hasattr(file_source, "read"):
        # io.BytesIO, file object, Streamlit UploadedFile...
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        data = file_source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        name = str(getattr(file_source, "name", "") or "").lower()
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        return bytes(data), name

    raise ValueError(f"Kiểu file_source không được hỗ trợ: {type(file_source).__name__}")


def _parse_csv_text(text: str) -> np.ndarray:
    """Tự parse văn bản CSV thành ma trận float64 2D (kiểm tra số cột đồng đều)."""
    rows: List[List[float]] = []
    delimiter = None
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        if delimiter is None:
            delimiter = next((d for d in (",", ";", "\t") if d in line), " ")
        tokens = [tok.strip() for tok in (line.split(delimiter) if delimiter != " " else line.split())]
        if tokens and tokens[-1] == "":
            tokens = tokens[:-1]  # dấu phẩy thừa cuối dòng
        try:
            values = [float(tok) for tok in tokens]
        except ValueError:
            if not rows:
                continue  # dòng header chữ ở đầu file -> bỏ qua
            raise ValueError(f"CSV dòng {line_no}: chứa giá trị không phải số: {raw[:80]!r}")
        if rows and len(values) != len(rows[0]):
            raise ValueError(
                f"CSV dòng {line_no}: có {len(values)} cột, khác {len(rows[0])} cột của các dòng trước."
            )
        rows.append(values)

    if not rows or not rows[0]:
        raise ValueError("File CSV rỗng hoặc không chứa ma trận số hợp lệ.")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("File CSV chứa giá trị NaN/inf.")
    return matrix


def _float_matrix_to_uint8(matrix: np.ndarray) -> np.ndarray:
    lo, hi = float(matrix.min()), float(matrix.max())
    if lo < 0.0 or hi > 255.0:
        warnings.warn(
            f"Ma trận CSV có giá trị ngoài [0, 255] (min={lo:g}, max={hi:g}); đã cắt (clip) về [0, 255].",
            RuntimeWarning,
            stacklevel=3,
        )
    return np.clip(np.round(matrix), 0, 255).astype(np.uint8)


def _resize_keep_aspect(pil_img: Image.Image, max_size: int) -> Image.Image:
    w, h = pil_img.size
    if max_size is None or max_size <= 0 or max(w, h) <= max_size:
        return pil_img
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return pil_img.resize(new_size, Image.Resampling.BILINEAR)


def _matrix_to_csv_text(matrix_2d: np.ndarray) -> str:
    if not isinstance(matrix_2d, np.ndarray) or matrix_2d.ndim != 2:
        raise ValueError(
            f"matrix_2d phải là mảng 2 chiều (H, W), nhận được shape {getattr(matrix_2d, 'shape', None)}."
        )
    if matrix_2d.size == 0:
        raise ValueError("matrix_2d rỗng.")

    arr = matrix_2d
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        fmt = "{:d}"
        arr = arr.astype(np.int64)
    else:
        arr = arr.astype(np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("matrix_2d chứa NaN/inf, không thể ghi CSV.")
        if np.all(arr == np.round(arr)):
            fmt = "{:d}"
            arr = arr.astype(np.int64)
        else:
            fmt = "{:.6g}"

    lines = [",".join(fmt.format(v) for v in row) for row in arr.tolist()]
    return "\n".join(lines) + "\n"


# =============================================================================
# API CÔNG KHAI
# =============================================================================

def load_image_or_csv(
    file_source: Union[str, Path, Any],
    max_size: int = 512,
) -> np.ndarray:
    """Đọc ảnh từ đường dẫn file hoặc đối tượng upload và chuẩn hóa về mảng NumPy uint8.

    Hỗ trợ đọc các định dạng ảnh tiêu chuẩn (.jpg, .jpeg, .png, .bmp, .tif, .webp) và file ma trận
    điểm ảnh (.csv). Tự động co dãn (downscale) bảo toàn tỷ lệ khung hình nếu
    kích thước chiều lớn nhất vượt quá `max_size`.

    Args:
        file_source: Đường dẫn tới file ảnh/CSV (str hoặc Path), bytes, hoặc đối tượng buffer
            nhị phân (ví dụ: `UploadedFile` từ Streamlit hoặc `io.BytesIO`).
            Nếu buffer không có tên file, hàm thử giải mã ảnh trước rồi mới thử CSV.
            Một `np.ndarray` (H, W) hoặc (H, W, 3) cũng được chấp nhận và chuẩn hóa về uint8.
        max_size: Kích thước tối đa của chiều dài hoặc chiều rộng sau khi đọc. Mặc định là 512.
            Nếu max(H, W) > max_size, ảnh (và ma trận CSV) sẽ được resize song tuyến (bilinear)
            bảo toàn tỷ lệ. Truyền `max_size <= 0` hoặc None để giữ nguyên kích thước.

    Returns:
        np.ndarray:
            - Nếu là ảnh thông thường: Mảng 3 chiều không gian màu RGB, kích thước `(H, W, 3)`,
              kiểu dữ liệu `np.uint8`, giá trị trong đoạn `[0, 255]`.
            - Nếu là ma trận xám từ CSV: Mảng 2 chiều kích thước `(H, W)`, kiểu `np.uint8`.

    Raises:
        ValueError: Nếu định dạng file không được hỗ trợ hoặc dữ liệu ma trận không hợp lệ.
        FileNotFoundError: Nếu file_source là đường dẫn không tồn tại trên hệ thống.

    Note:
        - Ảnh màu luôn trả về thứ tự kênh RGB (không phải BGR của OpenCV).
        - Ảnh chụp điện thoại được xoay đúng chiều theo thẻ EXIF Orientation.
        - Ảnh có kênh alpha (RGBA) được ghép lên nền trắng trước khi bỏ alpha.
        - CSV: giá trị được làm tròn; giá trị ngoài [0, 255] bị clip kèm cảnh báo RuntimeWarning.
    """
    if isinstance(file_source, np.ndarray):
        arr = file_source
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3)):
            raise ValueError(f"Mảng đầu vào phải có shape (H, W) hoặc (H, W, 3), nhận được {arr.shape}.")
        if arr.dtype != np.uint8:
            arr = np.clip(np.round(arr.astype(np.float64)), 0, 255).astype(np.uint8)
        return arr.copy()

    data, name = _read_source_bytes(file_source)
    if len(data) == 0:
        raise ValueError("Dữ liệu đầu vào rỗng.")
    ext = Path(name).suffix if name else ""

    if ext and ext not in SUPPORTED_CSV_EXTS and ext not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(
            f"Định dạng '{ext}' không được hỗ trợ. Hỗ trợ: "
            f"{sorted(SUPPORTED_IMAGE_EXTS | SUPPORTED_CSV_EXTS)}."
        )

    if ext in SUPPORTED_CSV_EXTS:
        return load_csv_matrix(data, max_size=max_size)

    try:
        with Image.open(io.BytesIO(data)) as pil_img:
            pil_img.load()
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                rgba = pil_img.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                pil_img = Image.alpha_composite(background, rgba)
            pil_img = pil_img.convert("RGB")
            pil_img = _resize_keep_aspect(pil_img, max_size)
            return np.asarray(pil_img, dtype=np.uint8).copy()
    except (OSError, Image.UnidentifiedImageError) as exc:
        if ext in SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"Không giải mã được file ảnh '{name}': {exc}") from exc
        # Buffer không tên: thử CSV
        try:
            return load_csv_matrix(data, max_size=max_size)
        except (UnicodeDecodeError, ValueError):
            raise ValueError("Dữ liệu không phải ảnh hợp lệ cũng không phải ma trận CSV hợp lệ.") from exc


def load_csv_matrix(
    file_source: Union[str, Path, bytes, Any],
    max_size: int = 0,
) -> np.ndarray:
    """Đọc file CSV ma trận điểm ảnh 2D thành mảng `(H, W)` `np.uint8`.

    Args:
        file_source: Đường dẫn file .csv, bytes hoặc buffer.
        max_size: Nếu > 0 và max(H, W) > max_size thì resize bilinear giữ tỷ lệ. Mặc định 0 (giữ nguyên).

    Returns:
        np.ndarray: `(H, W)` `np.uint8`.
    """
    data = file_source if isinstance(file_source, bytes) else _read_source_bytes(file_source)[0]
    text = data.decode("utf-8-sig")
    matrix_u8 = _float_matrix_to_uint8(_parse_csv_text(text))
    if max_size and max_size > 0 and max(matrix_u8.shape) > max_size:
        pil_img = _resize_keep_aspect(Image.fromarray(matrix_u8), max_size)
        matrix_u8 = np.asarray(pil_img, dtype=np.uint8).copy()
    return matrix_u8


def save_matrix_to_csv(
    matrix_2d: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Lưu ma trận 2D số nguyên (mức xám, nhãn phân cụm, hoặc mặt nạ) ra file CSV.

    Args:
        matrix_2d: Mảng 2 chiều kích thước `(H, W)`, kiểu số nguyên (`uint8` hoặc `int32`),
            đại diện cho giá trị mức xám, nhãn cụm K-means hoặc binary mask.
            Mảng bool được ghi thành 0/1; mảng float có phần thập phân được ghi dạng `%.6g`.
        output_path: Đường dẫn file đích lưu trữ (.csv). Thư mục cha được tạo tự động.

    Raises:
        ValueError: Nếu `matrix_2d` không phải mảng 2 chiều (ndim != 2) hoặc chứa NaN/inf.
    """
    text = _matrix_to_csv_text(matrix_2d)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")


def matrix_to_csv_bytes(matrix_2d: np.ndarray) -> bytes:
    """Chuyển đổi ma trận 2D thành chuỗi bytes định dạng CSV để phục vụ tải xuống trên Streamlit.

    Args:
        matrix_2d: Mảng 2 chiều kích thước `(H, W)` với kiểu dữ liệu số nguyên.

    Returns:
        bytes: Dữ liệu file CSV được mã hóa UTF-8 bytes.

    Raises:
        ValueError: Nếu `matrix_2d` không phải mảng 2 chiều.
    """
    return _matrix_to_csv_text(matrix_2d).encode("utf-8")
