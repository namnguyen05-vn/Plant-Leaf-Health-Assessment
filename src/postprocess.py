"""Binary Morphology and Connected Components Post-processing Module.

Module thực hiện các thuật toán hậu xử lý hình thái học nhị phân và phân tích liên thông từ đầu bằng NumPy/Python:
1. Phép co (Erosion) và phép dãn (Dilation) nhị phân với phần tử cấu trúc vuông.
2. Phép mở (Opening: Co trước -> Dãn sau) khử nhiễu đốm nhỏ.
3. Phép đóng (Closing: Dãn trước -> Co sau) lấp đầy lỗ hổng bên trong vết bệnh.
4. Thuật toán tìm thành phần liên thông 8 hướng bằng BFS (Breadth-First Search) để:
   - Loại bỏ các ổ nhiễu có diện tích nhỏ hơn ngưỡng `min_area`.
   - Đếm chính xác số lượng ổ bệnh độc lập trên bề mặt lá.
   - Trích xuất hộp bao chữ nhật Bounding Box (x, y, w, h, area) cho từng ổ bệnh.
5. Tiện ích vẽ Bounding Box và đánh số thứ tự ổ bệnh trực tiếp lên ảnh RGB bằng mảng NumPy.

QUY TẮC BẮT BUỘC (CODE FROM SCRATCH):
- TUYỆT ĐỐI KHÔNG sử dụng `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`, `cv2.connectedComponentsWithStats`.
- Sử dụng `np.pad` kết hợp `np.lib.stride_tricks.sliding_window_view` cho các phép hình thái học.
- Cài đặt thuật toán BFS hàng đợi (Queue) thuần Python kết hợp mảng kiểm tra đã thăm (visited array) NumPy.
"""

from collections import deque
from typing import Any, Dict, List, Tuple
import numpy as np


# Bảng phông chữ bitmap 5x7 tối giản (chỉ bằng NumPy) để vẽ chỉ số ổ bệnh #1, #2...
_DIGIT_FONT_5X7: Dict[str, Tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
}


def _validate_binary_mask_and_kernel(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Kiểm tra tính hợp lệ của mặt nạ 2D và kích thước kernel, chuẩn hóa về {0, 255} uint8."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"Mặt nạ đầu vào phải là mảng NumPy 2 chiều (H, W), nhận được: {getattr(mask, 'shape', type(mask))}.")
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Kích thước kernel phải là số nguyên dương lẻ (>= 1), nhận được: {kernel_size}.")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def erode_scratch(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Thực hiện phép co nhị phân (Binary Erosion) từ đầu bằng NumPy.

    Điểm ảnh tại vị trí (y, x) giữ giá trị 255 khi và chỉ khi TOÀN BỘ các điểm ảnh
    trong vùng lân cận kích thước `(kernel_size, kernel_size)` đều có giá trị 255:
        E(y, x) = 255 if all(N(y, x) == 255) else 0

    Args:
        mask: Mảng 2 chiều nhị phân kích thước `(H, W)`, kiểu `np.uint8`, chỉ nhận `{0, 255}`.
        kernel_size: Kích thước phần tử cấu trúc vuông (số lẻ >= 3, ví dụ: 3, 5). Mặc định là 3.

    Returns:
        np.ndarray: Mảng nhị phân sau co, kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.

    Raises:
        ValueError: Nếu `mask` không phải 2 chiều hoặc `kernel_size` không phải số lẻ.
    """
    bin_mask = _validate_binary_mask_and_kernel(mask, kernel_size)
    if kernel_size == 1:
        return bin_mask

    pad_w = kernel_size // 2
    padded = np.pad(bin_mask, pad_width=pad_w, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    eroded = np.min(windows, axis=(2, 3))
    return eroded.astype(np.uint8)


def dilate_scratch(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Thực hiện phép dãn nhị phân (Binary Dilation) từ đầu bằng NumPy.

    Điểm ảnh tại vị trí (y, x) nhận giá trị 255 nếu CÓ ÍT NHẤT MỘT điểm ảnh
    trong vùng lân cận `(kernel_size, kernel_size)` mang giá trị 255:
        D(y, x) = 255 if any(N(y, x) == 255) else 0

    Args:
        mask: Mảng 2 chiều nhị phân kích thước `(H, W)`, kiểu `np.uint8`, chỉ nhận `{0, 255}`.
        kernel_size: Kích thước phần tử cấu trúc vuông (số lẻ >= 3). Mặc định là 3.

    Returns:
        np.ndarray: Mảng nhị phân sau dãn, kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.

    Raises:
        ValueError: Nếu `mask` không phải 2 chiều hoặc `kernel_size` không hợp lệ.
    """
    bin_mask = _validate_binary_mask_and_kernel(mask, kernel_size)
    if kernel_size == 1:
        return bin_mask

    pad_w = kernel_size // 2
    padded = np.pad(bin_mask, pad_width=pad_w, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    dilated = np.max(windows, axis=(2, 3))
    return dilated.astype(np.uint8)


def opening_scratch(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Phép mở hình thái học (Morphological Opening): Co trước, Dãn sau.

    Mục đích: Loại bỏ các đốm nhiễu li ti, các nhánh mảnh gai của vết bệnh mà không làm thay đổi kích thước tổng thể.
        Opening(A, B) = Dilate(Erode(A, B), B)

    Args:
        mask: Mảng nhị phân `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.
        kernel_size: Kích thước phần tử cấu trúc. Mặc định là 3.

    Returns:
        np.ndarray: Mặt nạ sau phép mở, kích thước `(H, W)`, kiểu `np.uint8`.
    """
    eroded = erode_scratch(mask, kernel_size=kernel_size)
    return dilate_scratch(eroded, kernel_size=kernel_size)


def closing_scratch(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Phép đóng hình thái học (Morphological Closing): Dãn trước, Co sau.

    Mục đích: Lấp đầy các lỗ thủng nhỏ bên trong vết bệnh và nối liền các đoạn đứt gãy lân cận.
        Closing(A, B) = Erode(Dilate(A, B), B)

    Args:
        mask: Mảng nhị phân `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.
        kernel_size: Kích thước phần tử cấu trúc. Mặc định là 3.

    Returns:
        np.ndarray: Mặt nạ sau phép đóng, kích thước `(H, W)`, kiểu `np.uint8`.
    """
    dilated = dilate_scratch(mask, kernel_size=kernel_size)
    return erode_scratch(dilated, kernel_size=kernel_size)


def connected_components_bfs_scratch(
    mask: np.ndarray,
    min_area: int = 20,
) -> Dict[str, Any]:
    """Tìm các thành phần liên thông 8 hướng (8-connectivity) bằng BFS từ đầu, lọc nhiễu diện tích và xuất BBox.

    Thuật toán duyệt:
    1. Quét các điểm ảnh tiền cảnh (`mask > 0`) chưa được thăm.
    2. Dùng hàng đợi BFS (`collections.deque`) mở rộng tìm kiếm 8 láng giềng `(-1..1, -1..1)`.
    3. Chỉ giữ lại các thành phần liên thông có diện tích `area >= min_area`.
    4. Sắp xếp danh sách các ổ bệnh theo diện tích giảm dần.

    Args:
        mask: Mặt nạ nhị phân 2 chiều kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.
        min_area: Ngưỡng diện tích điểm ảnh tối thiểu để công nhận là một vết bệnh hợp lệ (mặc định 20 pixel).

    Returns:
        Dict[str, Any]: Dictionary chuẩn chứa các thông tin:
            - "cleaned_mask": np.ndarray kích thước `(H, W)`, kiểu `np.uint8`, chỉ nhận `{0, 255}`,
              chứa các vết bệnh đã được khử sạch nhiễu hạt nhỏ.
            - "num_lesions": int, số lượng ổ bệnh độc lập đếm được.
            - "bboxes": List[Tuple[int, int, int, int, int]], danh sách các hộp bao của từng vết bệnh
              được sắp xếp theo diện tích giảm dần, mỗi phần tử là tuple `(x, y, w, h, area)`.

    Raises:
        ValueError: Nếu `mask` không phải mảng 2 chiều hoặc `min_area < 1`.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"Mặt nạ đầu vào phải là mảng 2 chiều (H, W), nhận được: {getattr(mask, 'shape', type(mask))}.")
    if min_area < 1:
        raise ValueError(f"min_area phải >= 1, nhận được: {min_area}.")

    binary = mask > 0
    H, W = binary.shape
    visited = np.zeros((H, W), dtype=bool)
    cleaned_mask = np.zeros((H, W), dtype=np.uint8)
    bboxes: List[Tuple[int, int, int, int, int]] = []

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    fg_rows, fg_cols = np.where(binary)
    for start_r, start_c in zip(fg_rows.tolist(), fg_cols.tolist()):
        if visited[start_r, start_c]:
            continue

        visited[start_r, start_c] = True
        queue = deque([(start_r, start_c)])
        comp_rows: List[int] = []
        comp_cols: List[int] = []

        min_r = max_r = start_r
        min_c = max_c = start_c

        while queue:
            r, c = queue.popleft()
            comp_rows.append(r)
            comp_cols.append(c)

            if r < min_r:
                min_r = r
            elif r > max_r:
                max_r = r
            if c < min_c:
                min_c = c
            elif c > max_c:
                max_c = c

            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and binary[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        area = len(comp_rows)
        if area >= min_area:
            cleaned_mask[comp_rows, comp_cols] = 255
            x = int(min_c)
            y = int(min_r)
            w = int(max_c - min_c + 1)
            h = int(max_r - min_r + 1)
            bboxes.append((x, y, w, h, int(area)))

    # Sắp xếp danh sách các ổ bệnh theo diện tích giảm dần
    bboxes.sort(key=lambda item: item[4], reverse=True)

    return {
        "cleaned_mask": cleaned_mask,
        "num_lesions": int(len(bboxes)),
        "bboxes": bboxes,
    }


def fill_holes_bfs_scratch(mask: np.ndarray) -> np.ndarray:
    """Lấp đầy toàn bộ lỗ thủng bên trong vùng tiền cảnh (Binary Hole Filling) bằng BFS từ viền ảnh.

    Nguyên lý: Khởi tạo BFS 4-hướng từ tất cả các pixel nền (0) nằm trên 4 cạnh viền ảnh.
    Mọi pixel nền không thể chạm tới từ viền ngoài đều là lỗ thủng nằm kín bên trong đối tượng
    (ví dụ: đốm bệnh nâu/vàng hoặc vùng chói nắng bên trong phiến lá) và sẽ được lấp đầy thành 255.

    Args:
        mask: Mặt nạ nhị phân 2 chiều kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.

    Returns:
        np.ndarray: Mặt nạ đã lấp đầy lỗ thủng, kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.

    Raises:
        ValueError: Nếu `mask` không phải mảng 2 chiều.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(
            f"Mặt nạ đầu vào phải là mảng NumPy 2 chiều (H, W), nhận được: {getattr(mask, 'shape', type(mask))}."
        )
    H, W = mask.shape
    if H == 0 or W == 0:
        return mask.astype(np.uint8, copy=True)

    fg = (mask > 0).tolist()
    reachable_bg = [[False] * W for _ in range(H)]
    queue: deque = deque()

    # Thu thập toàn bộ pixel nền (== 0) nằm trên 4 cạnh viền ảnh
    for c in range(W):
        if not fg[0][c] and not reachable_bg[0][c]:
            reachable_bg[0][c] = True
            queue.append((0, c))
        if not fg[H - 1][c] and not reachable_bg[H - 1][c]:
            reachable_bg[H - 1][c] = True
            queue.append((H - 1, c))
    for r in range(1, H - 1):
        if not fg[r][0] and not reachable_bg[r][0]:
            reachable_bg[r][0] = True
            queue.append((r, 0))
        if not fg[r][W - 1] and not reachable_bg[r][W - 1]:
            reachable_bg[r][W - 1] = True
            queue.append((r, W - 1))

    # BFS 4-hướng trên vùng nền ngoài để đảm bảo không rò qua biên chéo của lá
    while queue:
        r, c = queue.popleft()
        if r > 0 and not fg[r - 1][c] and not reachable_bg[r - 1][c]:
            reachable_bg[r - 1][c] = True
            queue.append((r - 1, c))
        if r + 1 < H and not fg[r + 1][c] and not reachable_bg[r + 1][c]:
            reachable_bg[r + 1][c] = True
            queue.append((r + 1, c))
        if c > 0 and not fg[r][c - 1] and not reachable_bg[r][c - 1]:
            reachable_bg[r][c - 1] = True
            queue.append((r, c - 1))
        if c + 1 < W and not fg[r][c + 1] and not reachable_bg[r][c + 1]:
            reachable_bg[r][c + 1] = True
            queue.append((r, c + 1))

    bg_arr = np.asarray(reachable_bg, dtype=bool)
    return np.where(~bg_arr, 255, 0).astype(np.uint8)


def keep_largest_connected_component_scratch(mask: np.ndarray) -> np.ndarray:
    """Giữ lại duy nhất thành phần liên thông 8-hướng có diện tích lớn nhất (chiếc lá chính).

    Loại bỏ hoàn toàn các đốm cỏ dại, nhiễu nền đất/gỗ rời rạc nằm xung quanh rìa ảnh.

    Args:
        mask: Mặt nạ nhị phân 2 chiều kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.

    Returns:
        np.ndarray: Mặt nạ chỉ chứa thành phần liên thông lớn nhất, kích thước `(H, W)`, kiểu `np.uint8`.

    Raises:
        ValueError: Nếu `mask` không phải mảng 2 chiều.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(
            f"Mặt nạ đầu vào phải là mảng NumPy 2 chiều (H, W), nhận được: {getattr(mask, 'shape', type(mask))}."
        )
    H, W = mask.shape
    binary = mask > 0
    fg_rows, fg_cols = np.where(binary)
    if fg_rows.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    grid = binary.tolist()
    visited = [[False] * W for _ in range(H)]
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    best_rows: List[int] = []
    best_cols: List[int] = []

    for start_r, start_c in zip(fg_rows.tolist(), fg_cols.tolist()):
        if visited[start_r][start_c]:
            continue
        visited[start_r][start_c] = True
        queue = deque([(start_r, start_c)])
        comp_rows: List[int] = []
        comp_cols: List[int] = []

        while queue:
            r, c = queue.popleft()
            comp_rows.append(r)
            comp_cols.append(c)
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        if len(comp_rows) > len(best_rows):
            best_rows = comp_rows
            best_cols = comp_cols

    out = np.zeros((H, W), dtype=np.uint8)
    if best_rows:
        out[best_rows, best_cols] = 255
    return out


def _draw_text_numpy(
    canvas: np.ndarray,
    text: str,
    top_y: int,
    left_x: int,
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (220, 38, 38),
    scale: int = 1,
) -> None:
    """Vẽ chuỗi ký tự số đơn giản (#1, #2, ...) trực tiếp bằng mảng NumPy (không dùng OpenCV)."""
    H, W = canvas.shape[:2]
    char_w = 5 * scale
    char_h = 7 * scale
    pad = 2 * scale
    total_w = len(text) * (char_w + scale) - scale + 2 * pad
    total_h = char_h + 2 * pad

    y0 = max(0, top_y)
    x0 = max(0, left_x)
    y1 = min(H, y0 + total_h)
    x1 = min(W, x0 + total_w)

    if y1 <= y0 or x1 <= x0:
        return

    # Vẽ nền nhãn nhỏ gọn
    canvas[y0:y1, x0:x1] = bg_color

    cursor_x = x0 + pad
    cursor_y = y0 + pad
    for ch in text:
        glyph = _DIGIT_FONT_5X7.get(ch)
        if glyph is not None:
            for gy, row_bits in enumerate(glyph):
                for gx, bit in enumerate(row_bits):
                    if bit == "1":
                        py0 = cursor_y + gy * scale
                        px0 = cursor_x + gx * scale
                        py1 = min(H, py0 + scale)
                        px1 = min(W, px0 + scale)
                        if py0 < H and px0 < W and py1 > py0 and px1 > px0:
                            canvas[py0:py1, px0:px1] = fg_color
        cursor_x += char_w + scale


def draw_lesion_bboxes_scratch(
    rgb_img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, int]],
    box_color: Tuple[int, int, int] = (239, 68, 68),
    thickness: int = 2,
    draw_labels: bool = True,
) -> np.ndarray:
    """Vẽ hộp bao Bounding Box và số thứ tự ổ bệnh trực tiếp lên ảnh RGB bằng mảng NumPy.

    Args:
        rgb_img: Ảnh RGB gốc kích thước `(H, W, 3)` hoặc ảnh xám `(H, W)`, kiểu `np.uint8`.
        bboxes: Danh sách các hộp bao `(x, y, w, h, area)` từ `connected_components_bfs_scratch`.
        box_color: Màu RGB cho đường viền hộp bao. Mặc định là đỏ `(239, 68, 68)`.
        thickness: Độ dày đường viền (pixel). Mặc định là 2.
        draw_labels: Nếu True, vẽ số thứ tự `#1`, `#2`, ... ở góc trên trái mỗi BBox.

    Returns:
        np.ndarray: Bản sao ảnh RGB kích thước `(H, W, 3)` kiểu `np.uint8` đã được đánh dấu các ổ bệnh.
    """
    if rgb_img.ndim == 2:
        canvas = np.stack([rgb_img, rgb_img, rgb_img], axis=-1).astype(np.uint8)
    else:
        canvas = rgb_img.copy().astype(np.uint8)

    H, W = canvas.shape[:2]
    scale = 1 if max(H, W) < 400 else 2

    for idx, bbox in enumerate(bboxes, start=1):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1 = max(0, min(W - 1, x))
        y1 = max(0, min(H - 1, y))
        x2 = max(0, min(W - 1, x + w - 1))
        y2 = max(0, min(H - 1, y + h - 1))

        # Vẽ 4 cạnh hộp bao bằng NumPy slicing
        canvas[y1 : min(H, y1 + thickness), x1 : x2 + 1] = box_color
        canvas[max(0, y2 - thickness + 1) : y2 + 1, x1 : x2 + 1] = box_color
        canvas[y1 : y2 + 1, x1 : min(W, x1 + thickness)] = box_color
        canvas[y1 : y2 + 1, max(0, x2 - thickness + 1) : x2 + 1] = box_color

        if draw_labels:
            label_text = f"#{idx}"
            label_h = 7 * scale + 4 * scale
            label_y = y1 - label_h if y1 >= label_h else y1
            _draw_text_numpy(
                canvas,
                text=label_text,
                top_y=label_y,
                left_x=x1,
                fg_color=(255, 255, 255),
                bg_color=box_color,
                scale=scale,
            )

    return canvas
