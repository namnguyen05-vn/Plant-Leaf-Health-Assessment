# Ảnh lá thật `data/raw/real/` (đánh giá định tính)

22 ảnh, KHÔNG có mặt nạ Ground Truth. Mức bệnh trong `metadata_real.csv` được **ước lượng bằng mắt**.

| Nhóm (`group`) | Số ảnh | Nội dung |
|---|---|---|
| `healthy` | 3 | Lá cà chua khỏe (PlantVillage) |
| `lesion` | 6 | Lá cà chua có vết hoại tử/đốm nâu rõ, cấp 0–3 |
| `yellowing` | 2 | Lá cà chua vàng úa lan rộng |
| `mottle_shadow` | 7 | Lá cà chua khảm/úa loang, phần lớn có bóng đổ gắt |
| `failure` | 4 | Failure case cho báo cáo (bắp cải, giữ từ bộ cũ): nền gỗ, lá vàng toàn phần ngoài ruộng, đốm tím, lá kín khung |

Ảnh `tom_*.png` giữ nguyên định dạng PNG gốc (256×256), không chỉnh sửa.
Ảnh `fail_*` đã cắt ô số Bugwood (28 px ở đáy) hoặc thu nhỏ về 1280 px; không chỉnh màu.

## Nguồn & giấy phép
- **PlantVillage** (Hughes & Salathé, 2015, arXiv:1511.08060): bản phân phối trên Mendeley Data theo
  giấy phép CC0 1.0. Vẫn nên ghi nguồn trong báo cáo.
- `Bugwood.org #<số>`: giấy phép Creative Commons, bắt buộc ghi nguồn Bugwood.org.
- `Internet (không rõ tác giả)`: chỉ dùng minh họa trong bài tập.

## Chạy
```bash
python -m experiments.run_real_images   # -> experiments/results/real/
```
Thêm ảnh: chép vào thư mục này và thêm 1 dòng vào `metadata_real.csv`.
