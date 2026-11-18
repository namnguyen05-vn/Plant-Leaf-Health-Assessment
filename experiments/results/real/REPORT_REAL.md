# Đánh giá định tính trên ảnh lá thật (`data/raw/real/`)

Số liệu: `real_results.csv`, `summary.md`; hình: `real_results_grid.png`
(tái tạo: `python -m experiments.run_real_images`).

Bộ ảnh gồm **18 ảnh lá cà chua PlantVillage** (256×256, một lá trên nền xám đồng màu) và
**4 failure case** giữ lại từ bộ ảnh cũ. Ảnh thật không có Ground Truth: cấp độ "ước lượng" là đánh giá
bằng mắt, mang tính tham khảo.

## 1. Kết quả theo nhóm

| Nhóm | Ảnh | Tách lá | Phát hiện bệnh |
|---|---|---|---|
| Lá khỏe | 3 | 3/3 tốt | 2/3 đúng cấp 0. `tom_healthy_01` báo cấp 1 (R = 1.68%): báo giả ở **khe hẹp giữa hai lá chét**, nơi lẫn nền xám và bóng |
| Vết bệnh nâu rõ | 6 | 6/6 tốt | Cấp độ khớp 5/6, nhưng **R% chỉ khoảng một nửa** so với ước lượng (vd 5.7–6.3% so với 8–15%). `tom_lesion_02` bị bỏ sót (0.29%) |
| Vàng úa lan rộng | 2 | 2/2 tốt | **Đánh giá thấp nhiều**: 3.5–4% so với 20–35% |
| Khảm/úa loang + bóng đổ | 7 | **7/7 tốt, kể cả bóng đổ gắt** | Chỉ 0.2–3.8%. "Khớp" 5/7 chủ yếu vì khoảng ước lượng rộng (cấp 1–3) |

Kết luận chính:
1. **Tách nền lá (Tầng 1, Otsu) rất ổn định** trên nền đồng màu: 18/18 ảnh, bóng đổ đậm không bị
   nhận nhầm là lá. Đây là khác biệt lớn so với bộ ảnh cũ (nền gỗ, đất, ruộng, lá chạm viền ảnh).
2. **Hệ thống nhạy với hoại tử nâu, nhưng gần như bỏ qua vàng úa lan tỏa (chlorosis).** Vùng vàng có
   ranh giới mờ, chuyển dần sang xanh, nên bước so sánh với mô khỏe (z-score) và bộ lọc diện tích ổ bệnh
   tối thiểu chỉ giữ lại phần lõi đậm nhất. Đề xuất cho Vai trò 3: thử tăng độ nhạy quầng vàng hoặc thêm
   một kênh vàng (b* trong Lab) để bắt chlorosis.
3. Việc có tính vàng úa là "bệnh" hay không cũng còn tranh luận (ngay cả người đánh giá bằng mắt cũng
   khó vạch ranh giới). Trong báo cáo nên định nghĩa rõ: hệ thống đo **mô hoại tử và đốm bệnh có ranh giới**,
   chưa đo **úa vàng lan tỏa**.

## 2. Failure cases (4 ảnh, mỗi ảnh một nguyên nhân khác nhau)

| Ảnh | Ước lượng | Máy | Nguyên nhân |
|---|---|---|---|
| `fail_01_wood_deadtissue` | cấp 3 (40–50%) | cấp 1 (4.3%) | Mảng lá chết nâu xám **trùng màu nền gỗ**, nên bị loại khỏi mặt nạ lá như là nền. Khớp với kết quả bộ mô phỏng: nền gỗ có IoU thấp nhất |
| `fail_02_field_yellowed_lv4` | cấp 3–4 (50–70%) | cấp 0 (0%) | Lá vàng úa gần toàn bộ, không còn mô xanh làm mốc "khỏe" để so sánh; nền là lá xanh khác |
| `fail_03_bugwood5389805_purple_spots` | cấp 2 (15–25%) | cấp 1 (1.8%) | Đốm bệnh **màu xanh tím**, ngoài dải màu nâu/vàng mà phương pháp được thiết kế để nhận |
| `fail_04_bugwood1572866_fullframe` | cấp 1 | cấp 1 (3.8%) | "Khớp" nhưng **sai lý do**: lá phủ kín khung, viền ảnh không có nền nên chỉ một dải lá hẹp được nhận là lá. Minh họa vì sao chỉ đếm cấp độ có thể đánh lừa |

## 3. Khuyến nghị sử dụng (đưa vào báo cáo / app)
- Chụp **một lá**, trên **nền đồng màu tương phản** (xám, đen, trắng), lá cách viền ảnh khoảng 10%.
- Bóng đổ không phải vấn đề; **nền gỗ, nền đất, ruộng và ảnh chụp cận kín khung** thì có.
- Kết quả R% nên hiểu là **diện tích hoại tử/đốm bệnh rõ ranh giới**, có xu hướng thấp hơn đánh giá
  bằng mắt khi lá có vàng úa lan rộng.

## Nguồn
- Ảnh cà chua: PlantVillage — Hughes, D. P. & Salathé, M. (2015). *An open access repository of images on
  plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
- Failure case: xem `data/raw/real/metadata_real.csv`.
