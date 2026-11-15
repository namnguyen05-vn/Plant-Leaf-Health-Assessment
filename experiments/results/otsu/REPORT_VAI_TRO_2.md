# Báo cáo Vai trò 2 — Dữ liệu, Tiền xử lý & Thuật toán Otsu

Số liệu chi tiết: `summary.md`, `otsu_threshold_benchmark.csv`, `lesion_otsu_benchmark.csv`,
`speed_benchmark.csv` (tái tạo bằng `python -m experiments.otsu_analysis`).

## 1. Sản phẩm bàn giao

| Đầu ra (theo bảng phân công) | Vị trí trong repo | Trạng thái |
|---|---|---|
| Dataset 20–30 ảnh các mức độ + CSV mẫu | `data/raw/` (30 ảnh, GT lá + vết bệnh, 10 file CSV, `metadata.csv`) | Xong |
| I/O CSV (bảng ghi `file_handler.py`) | `src/data_io.py` | Xong |
| Chuyển màu (bảng ghi `color_space.py`) + lọc (bảng ghi `filters.py`) | `src/preprocess.py` | Xong |
| Otsu | `src/otsu.py` | Xong |
| T* lệch ≤ ±1 so với OpenCV | `tests/test_role2_preprocess_otsu.py` + thực nghiệm | **Đạt: lệch 0 ở 150/150 lần đo** |
| Biểu đồ Histogram & kết quả tách nền Otsu | `experiments/results/otsu/*.png` | Xong |

Ghi chú: bảng phân công ghi tên file `file_handler.py / color_space.py / filters.py`, nhưng repo
(và `pipeline.py`, `app.py`, `test_contracts.py`) dùng `data_io.py / preprocess.py`. Giữ đúng tên
trong repo để không làm gãy import của cả nhóm.

## 2. Thuật toán đã cài đặt (From Scratch, chỉ NumPy)

- **RGB → Gray**: Y = 0.299R + 0.587G + 0.114B (BT.601), tính trên float64 rồi làm tròn. Lệch OpenCV ≤ 1
  (OpenCV dùng số học dấu phẩy tĩnh 14-bit).
- **RGB → HSV**: H ∈ [0°, 360°) theo kênh cực đại, chia an toàn khi C = 0 hoặc V = 0; chuẩn hóa H, S, V về
  0–255 (tương đương `COLOR_RGB2HSV_FULL`).
- **Tích chập 2D**: đúng nghĩa toán học (lật kernel), đệm biên `reflect` = BORDER_REFLECT_101 của OpenCV,
  vector hóa kiểu "shift-and-add" (k² phép cộng mảng, bộ nhớ phụ O(H·W)).
- **Gauss / Mean**: tận dụng tính **tách được** (K = g ⊗ g) → 2 lần tích chập 1D, chi phí O(2k) thay vì
  O(k²) mỗi pixel. Có test chứng minh kết quả trùng tích chập 2D trực tiếp.
- **Otsu**: histogram 256 mức (`np.bincount`, hỗ trợ ROI) → ω0(t), μ(t) bằng `np.cumsum` →
  σ²_B(t) = (μ_T·ω0 − μ)² / (ω0·ω1) cho cả 256 ngưỡng cùng lúc (không vòng lặp theo t).
  Bổ sung so với khung ban đầu:
  - Chọn t **nhỏ nhất** trong các cực đại bằng nhau (dung sai tương đối 1e-9) — khớp OpenCV khi histogram
    có khoảng trống giữa 2 đỉnh (σ²_B phẳng). Không xử lý điểm này thì T* có thể lệch hàng chục mức.
  - Tham số `invert` cho đối tượng tối hơn nền; trả thêm `separability` η = σ²_B(T*) / σ²_T ∈ [0, 1].
  - Ảnh đồng nhất / ROI rỗng không ném lỗi mà trả T* = 0 (giống OpenCV) để pipeline không dừng.

## 3. Kết quả chính

### 3.1 Độ chính xác so với OpenCV
- Otsu: |ΔT*| = **0** ở cả 150 lần đo (30 ảnh × 4 kênh tách nền + 30 lần trong ROI lá), cộng thêm
  40 ảnh hỗn hợp Gauss ngẫu nhiên trong test.
- Gray ≤ 1 mức, HSV ≤ 1 mức (H tính vòng), Gaussian ≤ 1, Mean = 0 (trên ảnh 512×512).
- Tích hợp: khi thay hàm dự phòng của pipeline bằng code này, `leaf_mask` và chẩn đoán của
  **30/30 ảnh giống hệt từng pixel** → không gây hồi quy cho các vai trò khác.

### 3.2 Phân tích Histogram đa kênh (`histogram_pixel_classes.png`)
Đo độ chồng lấn Σ min(p, q) giữa phân bố của các lớp pixel GT (0 = tách hoàn toàn, 1 = trùng nhau):

| Kênh | Nền ↔ Lá khỏe | Lá khỏe ↔ Vết bệnh |
|---|---|---|
| Gray | 0.43 | 0.30 |
| S | 0.13 | **0.82** |
| H | **0.03** | **0.05** |
| ExG | **0.00** | 0.09 |

Kết luận:
1. **Gray là kênh kém nhất để tách nền**: lá xanh có độ sáng trung bình, tối hơn giấy trắng nhưng sáng hơn
   nền tối/đất → cực tính đảo theo nền và histogram chồng lấn mạnh.
2. **S (độ bão hòa) tách nền tốt nhưng gần như "mù" với vết bệnh** (chồng lấn khỏe–bệnh 0.82): lá xanh và
   đốm nâu/vàng đều bão hòa cao. → S phù hợp cho Tầng 1, không dùng được cho tách vết bệnh.
3. **H (Hue) tách mô khỏe với vết bệnh tốt nhất**: lá khỏe tập trung quanh H ≈ 70 (≈ 100°), đốm nâu/vàng ở
   H ≈ 15–45. Nhưng nền đất/gỗ cũng có Hue nâu → chỉ dùng H **bên trong ROI lá**, đúng thiết kế đa tầng.

### 3.3 Tầng 1 — Tách nền bằng Otsu (IoU trung bình so với GT, sau khi giữ thành phần lớn nhất)

| Kênh | IoU | Điểm yếu |
|---|---|---|
| Gray (tự chọn cực tính) | 0.589 | hỏng trên nền gỗ (0.14), đất (0.50), bóng đổ (0.21) |
| S (HSV) | 0.902 | hỏng trên nền **gỗ** (0.70; `moderate_05`, `critical_02` IoU ≈ 0) vì gỗ cũng bão hòa |
| ExG = 2g − r − b | 0.830 | gân vàng (`yellowvein_01` 0.12), lá bệnh nặng |
| VegMap (pipeline của nhóm trưởng) | **0.942** | ổn định nhất; thấp nhất 0.65 ở `critical_02` (lá úa > 65%) |

→ Xác nhận lựa chọn của pipeline: Otsu trên bản đồ thực vật (VegMap) tốt hơn Otsu trên Gray/S đơn lẻ.
Bóng đổ **không** làm hỏng S/VegMap (bóng giảm V nhưng ít ảnh hưởng S, Hue), nhưng làm Gray sụp (0.21).

### 3.4 Otsu cho đốm bệnh (Hue trong ROI lá GT)
- Lá có bệnh: IoU 0.72–0.92; R% bị **ước lượng thấp** đều đặn ~15–20% tương đối (VD severe 41.4% → 34.6%)
  vì phần ngoài của quầng vàng có Hue trung gian, bị xếp vào lớp "khỏe".
- **Hạn chế bản chất của Otsu**: luôn chia ảnh thành 2 lớp kể cả khi chỉ có 1 lớp. Trên lá khỏe, Otsu
  vẫn "tìm ra" 2–74% vết bệnh giả. Chỉ số η giúp phát hiện: lá khỏe η = 0.48–0.73, lá bệnh η ≥ 0.74.
  Một ngưỡng chặn kiểu η < ~0.75 ⇒ "không có vết bệnh" sẽ loại phần lớn báo động giả, **trừ**
  `yellowvein_01` (η = 0.875): gân vàng thật sự giống bệnh về màu → cần thêm tiêu chí hình học (gân là
  đường mảnh, dài; đốm bệnh là vùng tròn) ở bước hậu xử lý. Ngưỡng 0.75 được chọn trên dữ liệu mô phỏng,
  cần kiểm chứng lại trên ảnh thật trước khi dùng.

### 3.5 Tốc độ (ảnh 512×512)
Otsu tự code ~0.9 ms (OpenCV ~0.2 ms); Gaussian 5×5 ảnh màu ~16 ms; RGB→HSV ~41 ms (chậm nhất, vì nhiều
phép `np.where` trên float64). Tổng tiền xử lý + Otsu < 70 ms/ảnh — không phải điểm nghẽn của pipeline.

## 4. Gợi ý cho các vai trò khác (rút ra từ dataset có GT)
- Chạy toàn pipeline trên 30 ảnh: đúng cấp độ **25/30**. Lỗi chủ yếu do R% bị đánh giá thấp
  (`moderate_02`, `moderate_05` → cấp 1; `critical_01/02` → cấp 3) và gân vàng → cấp 1 (báo động giả).
  Có thể dùng `data/raw/ground_truth/*_lesion.png` để đo IoU/Dice trong `benchmark.py` (Vai trò 1) và chọn
  K / kênh màu (Vai trò 3).

## 5. Giới hạn cần nói rõ trong báo cáo
- Dataset là **ảnh mô phỏng**; số liệu IoU trên ảnh thật sẽ thấp hơn. Nên bổ sung 10–20 ảnh chụp thật
  (PlantVillage hoặc tự chụp) vào `data/raw/real/` để minh họa định tính.
- GT lá không gồm cuống lá, nên cuống bị tính là "nhận nhầm" (đỏ) trong ảnh lưới kết quả.
