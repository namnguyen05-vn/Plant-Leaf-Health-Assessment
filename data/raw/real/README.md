# Ảnh lá thật `data/raw/real/` (đánh giá định tính)

17 ảnh, KHÔNG có mặt nạ Ground Truth. Mức bệnh trong `metadata_real.csv` được **ước lượng bằng mắt**.

| Nhóm (`group`) | Số ảnh | Nội dung |
|---|---|---|
| `healthy` | 2 | Lá bắp cải khỏe tự chụp (lá trong, màu trắng kem) trên vải đen |
| `diseased` | 7 | Lá bắp cải/họ cải bệnh cấp 1–3, phần lớn nền đen |
| `hard` | 4 | Ca khó: nền đất + bóng đổ, ngoài ruộng, nền gỗ, lá vàng úa cấp 4 |
| `limitation` | 3 | Bệnh không biểu hiện bằng màu nâu/vàng (đốm tím, đốm trắng, khảm virus) |

Tên file: `cab_lv<cấp chính>_<stt>_<nguồn>[_ghi chú].jpg` (bắp cải), `lim_<stt>_...` (giới hạn).
`fills_frame = yes`: lá phủ kín khung hình, nên bước tách nền không đáng tin.

## Tiền xử lý đã áp dụng
- Ảnh Bugwood: cắt dải 28 px ở đáy để bỏ ô số ảnh đen trắng (tránh bị nhận nhầm là vật thể).
  Ảnh #1634015 cắt thêm viền khung phim.
- Ảnh tự chụp (2576×1932): cắt quanh lá (lề 22%) để bỏ vật lạ ở góc, rồi thu nhỏ về 1280 px.
- Ảnh lớn khác: thu nhỏ về cạnh dài tối đa 1280 px. Không chỉnh màu ảnh nào.

## Nguồn & giấy phép
- `Bugwood.org #<số>`: Bugwood Network Image Archives, giấy phép Creative Commons. Dùng cho giáo dục phi
  thương mại được, **bắt buộc ghi nguồn** (tên tác giả tra theo số ảnh tại images.bugwood.org).
- `Internet (không rõ tác giả)`: không còn link gốc; chỉ dùng minh họa trong bài tập, không phân phối lại.
- `Tự chụp`: ảnh của nhóm.

## Chạy
```bash
python -m experiments.run_real_images   # -> experiments/results/real/
```
Thêm ảnh: chép vào thư mục này và thêm 1 dòng vào `metadata_real.csv`.
