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
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


# Bảng phông chữ bitmap siêu gọn 3x5 (100% NumPy) để vẽ nhãn #1, #2... sắc nét không che vết bệnh
_MICRO_FONT_3X5: Dict[str, Tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "#": ("101", "111", "101", "111", "101"),
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


def keep_largest_connected_component_scratch(
    mask: np.ndarray,
    prefer_central: bool = True,
) -> np.ndarray:
    """Giữ lại thành phần liên thông 8-hướng chính của chiếc lá (ưu tiên thành phần lớn nằm gần tâm ảnh).

    Loại bỏ hoàn toàn các đốm cỏ dại, nhiễu nền hoặc vùng viền phông nền bám ở 4 góc ảnh.

    Args:
        mask: Mặt nạ nhị phân 2 chiều kích thước `(H, W)`, kiểu `np.uint8`, giá trị `{0, 255}`.
        prefer_central: Nếu True, ưu tiên trọng số theo khoảng cách tới tâm ảnh để tránh chọn
            nhầm mảng nền rìa ảnh. Mặc định là True.

    Returns:
        np.ndarray: Mặt nạ chỉ chứa thành phần liên thông chính, kích thước `(H, W)`, kiểu `np.uint8`.

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

    half_h = max(H / 2.0, 1.0)
    half_w = max(W / 2.0, 1.0)
    best_score = -1.0
    best_rows: List[int] = []
    best_cols: List[int] = []

    for start_r, start_c in zip(fg_rows.tolist(), fg_cols.tolist()):
        if visited[start_r][start_c]:
            continue
        visited[start_r][start_c] = True
        queue = deque([(start_r, start_c)])
        comp_rows: List[int] = []
        comp_cols: List[int] = []
        touches_top = touches_bot = touches_left = touches_right = False

        while queue:
            r, c = queue.popleft()
            comp_rows.append(r)
            comp_cols.append(c)
            if r == 0:
                touches_top = True
            if r == H - 1:
                touches_bot = True
            if c == 0:
                touches_left = True
            if c == W - 1:
                touches_right = True

            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        area = len(comp_rows)
        if prefer_central:
            cy = sum(comp_rows) / float(area)
            cx = sum(comp_cols) / float(area)
            dist_norm_sq = ((cy - half_h) / half_h) ** 2 + ((cx - half_w) / half_w) ** 2
            border_count = int(touches_top) + int(touches_bot) + int(touches_left) + int(touches_right)
            border_penalty = 0.30 if border_count >= 3 else 1.0
            score = float(area) * math.exp(-0.55 * dist_norm_sq) * border_penalty
        else:
            score = float(area)

        if score > best_score:
            best_score = score
            best_rows = comp_rows
            best_cols = comp_cols

    out = np.zeros((H, W), dtype=np.uint8)
    if best_rows:
        out[best_rows, best_cols] = 255
    return out


def filter_margin_noise_components_scratch(
    mask: np.ndarray,
    inner_mask: np.ndarray,
    min_area: int = 20,
    min_inner_ratio: float = 0.40,
) -> Dict[str, Any]:
    """Lọc thành phần liên thông BFS 8-hướng và xóa sạch nhiễu viền mép răng cưa.

    Chỉ giữ lại các ổ bệnh có diện tích `>= min_area` VÀ thỏa mãn điều kiện nằm bên trong lá:
    tâm hình học `(cy, cx)` nằm trong `inner_mask` HOẶC có ít nhất `min_inner_ratio` (mặc định 40%)
    diện tích nằm bên trong `inner_mask`.

    Args:
        mask: Mặt nạ vùng bệnh nhị phân `(H, W)`, kiểu `np.uint8` `{0, 255}`.
        inner_mask: Mặt nạ nội vùng lá đã co mép `(H, W)`, kiểu `np.uint8` `{0, 255}`.
        min_area: Diện tích điểm ảnh tối thiểu (mặc định 20).
        min_inner_ratio: Tỷ lệ diện tích tối thiểu nằm bên trong `inner_mask` (mặc định 0.40).

    Returns:
        Dict[str, Any]: Dictionary chuẩn `"cleaned_mask"`, `"num_lesions"`, `"bboxes"`.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(
            f"Mặt nạ đầu vào phải là mảng 2 chiều (H, W), nhận được: {getattr(mask, 'shape', type(mask))}."
        )
    if inner_mask.shape != mask.shape:
        raise ValueError(f"inner_mask {inner_mask.shape} không khớp kích thước mask {mask.shape}.")
    if min_area < 1:
        raise ValueError(f"min_area phải >= 1, nhận được: {min_area}.")

    if int(np.sum(inner_mask > 0)) == 0:
        return connected_components_bfs_scratch(mask, min_area=min_area)

    H, W = mask.shape
    binary = mask > 0
    fg_rows, fg_cols = np.where(binary)
    cleaned_mask = np.zeros((H, W), dtype=np.uint8)
    bboxes: List[Tuple[int, int, int, int, int]] = []

    if fg_rows.size == 0:
        return {"cleaned_mask": cleaned_mask, "num_lesions": 0, "bboxes": bboxes}

    grid = binary.tolist()
    inner_grid = (inner_mask > 0).tolist()
    visited = [[False] * W for _ in range(H)]
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for start_r, start_c in zip(fg_rows.tolist(), fg_cols.tolist()):
        if visited[start_r][start_c]:
            continue
        visited[start_r][start_c] = True
        queue = deque([(start_r, start_c)])
        comp_rows: List[int] = []
        comp_cols: List[int] = []
        min_r = max_r = start_r
        min_c = max_c = start_c
        inside_count = 0

        while queue:
            r, c = queue.popleft()
            comp_rows.append(r)
            comp_cols.append(c)
            if inner_grid[r][c]:
                inside_count += 1
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
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        area = len(comp_rows)
        if area < min_area:
            continue

        cy = min(H - 1, max(0, int(round(sum(comp_rows) / float(area)))))
        cx = min(W - 1, max(0, int(round(sum(comp_cols) / float(area)))))
        inside_ratio = inside_count / float(area)

        if inner_grid[cy][cx] or inside_ratio >= min_inner_ratio:
            cleaned_mask[comp_rows, comp_cols] = 255
            x = int(min_c)
            y = int(min_r)
            w = int(max_c - min_c + 1)
            h = int(max_r - min_r + 1)
            bboxes.append((x, y, w, h, int(area)))

    bboxes.sort(key=lambda item: item[4], reverse=True)
    return {
        "cleaned_mask": cleaned_mask,
        "num_lesions": int(len(bboxes)),
        "bboxes": bboxes,
    }


def _draw_text_numpy(
    canvas: np.ndarray,
    text: str,
    top_y: int,
    left_x: int,
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (220, 38, 38),
    scale: int = 1,
) -> None:
    """Vẽ chuỗi ký tự siêu gọn (#1, #2, ...) bằng font bitmap 3x5 hoàn toàn bằng NumPy."""
    H, W = canvas.shape[:2]
    char_w = 3 * scale
    char_h = 5 * scale
    gap = scale
    pad = max(1, scale)
    total_w = len(text) * (char_w + gap) - gap + 2 * pad
    total_h = char_h + 2 * pad

    x0 = max(0, min(W - total_w, left_x))
    y0 = max(0, min(H - total_h, top_y))
    x1 = min(W, x0 + total_w)
    y1 = min(H, y0 + total_h)

    if y1 <= y0 or x1 <= x0:
        return

    # Pha trộn nền nhãn bán trong suốt (alpha 0.85) để không che lấp chi tiết bên dưới
    bg_arr = np.array(bg_color, dtype=np.float32)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(np.round(0.85 * bg_arr + 0.15 * roi), 0, 255).astype(np.uint8)

    cursor_x = x0 + pad
    cursor_y = y0 + pad
    for ch in text:
        glyph = _MICRO_FONT_3X5.get(ch)
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
        cursor_x += char_w + gap


def draw_lesion_bboxes_scratch(
    rgb_img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, int]],
    box_color: Tuple[int, int, int] = (239, 68, 68),
    thickness: Optional[int] = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Vẽ hộp bao Bounding Box và số thứ tự ổ bệnh trực tiếp lên ảnh RGB bằng mảng NumPy.

    Độ dày nét vẽ và cỡ chữ tự động co giãn tỷ lệ theo kích thước ảnh để nhãn `#1, #2...`
    nhỏ gọn, sắc nét, đặt ở mép trên khung mà không che khuất vết bệnh bên dưới.

    Args:
        rgb_img: Ảnh RGB gốc kích thước `(H, W, 3)` hoặc ảnh xám `(H, W)`, kiểu `np.uint8`.
        bboxes: Danh sách các hộp bao `(x, y, w, h, area)` từ `connected_components_bfs_scratch`.
        box_color: Màu RGB cho đường viền hộp bao. Mặc định là đỏ `(239, 68, 68)`.
        thickness: Độ dày đường viền (pixel). Nếu None, tự tính `max(1, int(min(H, W) / 250))`.
        draw_labels: Nếu True, vẽ số thứ tự `#1`, `#2`, ... ở góc trên trái mỗi BBox.

    Returns:
        np.ndarray: Bản sao ảnh RGB kích thước `(H, W, 3)` kiểu `np.uint8` đã được đánh dấu các ổ bệnh.
    """
    if rgb_img.ndim == 2:
        canvas = np.stack([rgb_img, rgb_img, rgb_img], axis=-1).astype(np.uint8)
    else:
        canvas = rgb_img.copy().astype(np.uint8)

    H, W = canvas.shape[:2]
    min_dim = min(H, W)
    line_t = max(1, int(min_dim / 250)) if thickness is None else max(1, int(thickness))
    font_scale = max(0.35, min_dim / 900.0)
    int_scale = 1 if font_scale < 0.78 else max(2, int(round(font_scale * 1.4)))

    char_h = 5 * int_scale
    pad = max(1, int_scale)
    label_h = char_h + 2 * pad

    for idx, bbox in enumerate(bboxes, start=1):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1 = max(0, min(W - 1, x))
        y1 = max(0, min(H - 1, y))
        x2 = max(0, min(W - 1, x + w - 1))
        y2 = max(0, min(H - 1, y + h - 1))

        # Vẽ 4 cạnh hộp bao bằng NumPy slicing
        canvas[y1 : min(H, y1 + line_t), x1 : x2 + 1] = box_color
        canvas[max(0, y2 - line_t + 1) : y2 + 1, x1 : x2 + 1] = box_color
        canvas[y1 : y2 + 1, x1 : min(W, x1 + line_t)] = box_color
        canvas[y1 : y2 + 1, max(0, x2 - line_t + 1) : x2 + 1] = box_color

        if draw_labels:
            label_text = f"#{idx}"
            label_y = y1 - label_h if y1 >= label_h else y1
            _draw_text_numpy(
                canvas,
                text=label_text,
                top_y=label_y,
                left_x=x1,
                fg_color=(255, 255, 255),
                bg_color=box_color,
                scale=int_scale,
            )

    return canvas
