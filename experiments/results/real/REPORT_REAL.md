# Đánh giá định tính trên ảnh lá thật (17 ảnh, `data/raw/real/`)

Số liệu: `real_results.csv`, `summary.md`; hình: `real_results_grid.png`
(tái tạo: `python -m experiments.run_real_images`).

## 1. Cảnh báo về chỉ số "khớp cấp độ"

Nếu chỉ đếm cấp độ dự đoán nằm trong khoảng ước lượng bằng mắt, pipeline **khớp 11/14 ảnh bắp cải**.
Nhưng xem từng hình kết quả thì nhiều ca khớp **vì hai lỗi bù trừ nhau**: vùng lá bị tách sai (mất nửa lá
hoặc dính nền), đồng thời vết bệnh nhận sai chỗ, nên tỷ lệ R% vô tình rơi đúng khoảng. Vì vậy cần đánh
giá thêm bằng mắt cả hai bước:

| Ảnh | Cấp ước lượng → dự đoán | Tách lá | Nhận vết bệnh | Đánh giá thật |
|---|---|---|---|---|
| cab_lv0_01_selfshot | 0 → 0 | Tốt | Không báo giả | **Đúng** |
| cab_lv0_02_selfshot | 0 → 0 (0.97%) | Tốt | Báo giả ở đầu gân bị cắt | **Đúng**, sát ngưỡng 1% |
| cab_lv1_01_bugwood1572855 | 1 → 1 | Tốt | Bắt đúng các vết chữ V ở mép | **Đúng** (ca đẹp nhất) |
| cab_lv1_02_bugwood1572869 | 1 → 1 | **Mất nửa trái lá** | Bắt đúng đốm vòng lớn | Khớp nhờ may mắn |
| cab_lv1_03_bugwood1634015 | 1 → 1 | **Hỏng**: chỉ lấy góc dưới trái + cỏ | Một phần | Khớp nhờ may mắn |
| cab_lv1_04_..._fullframe | 1 → 1 | **Hỏng**: chỉ một dải hẹp | Gần như không | Khớp nhờ may mắn |
| cab_lv1_05_..._fullframe | 1–2 → 1 | Mất khoảng nửa lá | Bắt được nhiều đốm vòng | Một phần |
| cab_lv2_01_internet_blackbg | 1–2 → 1 | Tốt | Bắt đúng vết chữ V từ chóp lá | **Đúng** |
| cab_lv2_02_..._soil_shadow | 2 → 2 | Dính nền đất | **Nền đất bị tính là bệnh** | Khớp nhờ may mắn |
| cab_lv2_03_internet_field | 2 → 2 | Dính lá bên cạnh | Đất ở góc bị tính là bệnh | Khớp nhờ may mắn |
| cab_lv3_01_bugwood1534036 | 2–3 → 2 (20.4%) | Khá | Bắt phần lớn đốm | **Đúng** (hơi thấp) |
| cab_lv3_02_internet_blackbg | 2–3 → 1 (6.8%) | Tốt | **Bỏ sót mảng vàng nâu loang, tương phản thấp** | Sai (thấp hơn) |
| cab_lv3_03_internet_wood | 3 → 1 (4.3%) | **Mảng lá chết bị coi là nền gỗ** | — | Sai (thấp hơn) |
| cab_lv4_01_internet_field | 3–4 → 0 (0%) | Dính lá bên cạnh | **Không nhận vùng vàng úa và đốm xám** | Sai (thấp hơn nhiều) |

→ **Đúng thật sự: 6/14** (lv0_01, lv0_02, lv1_01, lv2_01, lv3_01, cộng lv1_05 đúng một phần).
Các ca đúng đều là **lá trọn vẹn trên nền đen/tối đồng màu**.

## 2. Nguyên nhân lỗi (kiểm chứng giả thuyết từ dữ liệu mô phỏng)

1. **Lá chạm viền ảnh → tách lá hỏng** (lv1_02, lv1_03, lv1_04, lv1_05). Pipeline ước lượng màu nền từ
   dải viền khung hình; khi viền là chính lá thì "nền" bị học thành màu lá.
2. **Mô chết màu nâu xám trùng màu nền gỗ** (lv3_03). Đúng như dự đoán từ dataset mô phỏng (nền gỗ có
   IoU thấp nhất): phần lá chết bị loại khỏi mặt nạ lá nên R% giảm từ khoảng 45% xuống 4%.
3. **Nền đất nâu trùng màu vết bệnh** (lv2_02, lv2_03): đất lọt vào mặt nạ lá rồi bị tính là bệnh.
4. **Vết bệnh tương phản thấp / lá vàng toàn phần** (lv3_02, lv4_01): khi phần lớn lá đã đổi màu, "mô
   khỏe" làm mốc so sánh không còn đủ, nên hệ thống không tách được vết bệnh. Khớp với hạn chế bản chất của
   Otsu đã phân tích ở `experiments/results/otsu/REPORT_VAI_TRO_2.md`.

## 3. Giới hạn của phương pháp dựa trên màu (nhóm `lim_*`)

| Ảnh | Kết quả | Nhận xét |
|---|---|---|
| lim_01 đốm xanh tím (bắp cải) | R = 1.8% (ước lượng khoảng 20%) | Bỏ sót như dự đoán: đốm không phải nâu/vàng |
| lim_02 đốm trắng (dưa chuột) | R = 2.6%, "khớp" | Khớp **sai lý do**: vùng đỏ nằm ở mép lá và khe cuống, không phải các đốm trắng |
| lim_03 khảm virus (dưa chuột) | R = 0.15% | Không có ranh giới vết bệnh rõ: nằm ngoài phạm vi bài toán |

## 4. Kết luận

- Pipeline hoạt động tốt với **ảnh chụp chuẩn**: một lá trọn vẹn, nền tối đồng màu, vết bệnh nâu/vàng rõ.
  Cả 2 lá khỏe tự chụp đều được chẩn đoán đúng cấp 0.
- Với ảnh ngoài đồng, ảnh chụp cận kín khung, nền đất/gỗ, hoặc lá bệnh rất nặng thì không đáng tin.
- **Khuyến nghị sử dụng** (đưa vào báo cáo/app): chụp 1 lá, đặt trên nền tối hoặc trắng đồng màu, để lá
  nằm trọn trong khung và cách viền ảnh ít nhất khoảng 10%.
- Số lượng ảnh ít (17) và cấp độ ước lượng bằng mắt, nên phần này chỉ là **đánh giá định tính**; số liệu
  định lượng nằm ở bộ mô phỏng có Ground Truth.
