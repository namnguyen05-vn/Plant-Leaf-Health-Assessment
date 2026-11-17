# Bộ dữ liệu chuẩn `data/raw/` (Vai trò 2)

30 ảnh lá cây có **mặt nạ Ground Truth** cho cả phiến lá và vết bệnh, phủ đủ 5 cấp độ bệnh của
`src/benchmark.diagnose_leaf_severity` và các điều kiện chụp khó.

Tái tạo y hệt (seed cố định): `python -m experiments.build_dataset`

## Cấu trúc

```text
data/raw/
├── <tên>.jpg                     # 30 ảnh RGB 512x384 hoặc 384x512, JPEG q=92
├── metadata.csv                  # nhãn của từng ảnh (xem bảng cột bên dưới)
├── ground_truth/
│   ├── <tên>_leaf.png            # mặt nạ phiến lá {0,255} (KHÔNG gồm cuống lá)
│   └── <tên>_lesion.png          # mặt nạ vết bệnh {0,255} = lõi hoại tử nâu + quầng vàng
└── csv/                          # file CSV mẫu (ma trận số nguyên, không header, dấu phẩy)
    ├── <tên>_gray_q4.csv         # ảnh mức xám thu nhỏ 1/4 (128x96) — mở được bằng Excel / app.py
    └── <tên>_leafmask_q4.csv     # mặt nạ lá GT tương ứng {0,255}
```

## Thành phần

| Nhóm | Số ảnh | Tỷ lệ bệnh GT (R%) | Cấp độ GT |
|---|---|---|---|
| `healthy_*` — lá khỏe | 5 | 0 | 0 |
| `mild_*` — bệnh nhẹ | 6 | 3.6 – 6.7 | 1 |
| `moderate_*` — trung bình | 5 | 13.1 – 22.6 | 2 |
| `severe_*` — nặng | 4 | 35.1 – 45.7 | 3 |
| `critical_*` — nghiêm trọng (có cháy mép lá) | 2 | 65.8 – 67.1 | 4 |
| `shadow_*` — bóng đổ cắt ngang lá | 2 | 5.2 / 19.7 | 1 / 2 |
| `glare_*` — chói sáng phản xạ | 2 | 0 / 6.6 | 0 / 1 |
| `noise_*` — nhiễu Gauss σ=18 + muối tiêu 1.2% | 2 | 6.7 / 17.9 | 1 / 2 |
| `yellowvein_01` — gân lá vàng (lá KHỎE) | 1 | 0 | 0 |
| `lowlight_01` — thiếu sáng (×0.42) | 1 | 9.1 | 1 |

5 loại nền xoay vòng: giấy trắng (`paper`), đất (`soil`), gỗ vân (`wood`), nền tối (`dark`), bàn xám (`gray`).

## Cột của `metadata.csv`

| Cột | Ý nghĩa |
|---|---|
| `filename`, `category`, `condition`, `background` | tên file, nhóm, điều kiện chụp, loại nền |
| `height`, `width` | kích thước ảnh |
| `leaf_area_px`, `lesion_area_px` | diện tích GT (pixel) |
| `gt_disease_ratio_pct` | R% = lesion / leaf × 100 |
| `gt_severity_level` | cấp 0–4 theo ngưỡng 1 / 10 / 25 / 50 % |
| `gt_num_lesions` | số ổ bệnh (thành phần liên thông 8-lân cận của mặt nạ lesion) |
| `seed` | seed sinh ảnh |

## Lưu ý khi dùng

- **Đây là ảnh mô phỏng (procedural), không phải ảnh chụp thật.** Lý do: ảnh thật công khai
  (vd. PlantVillage) không có mặt nạ vết bệnh, nên không tính được IoU / Dice / sai số R%.
  Ảnh mô phỏng cho phép đo định lượng chính xác; nhưng kết quả trên ảnh mô phỏng thường lạc quan hơn
  ảnh thật. Nên bổ sung ảnh chụp thật vào `data/raw/real/` để đánh giá định tính trong báo cáo.
- GT lá **không gồm cuống** (pipeline cũng cắt cuống khi tính diện tích phiến lá).
- GT vết bệnh **gồm cả quầng vàng** (mô úa vàng bao quanh đốm nâu cũng là mô bệnh).
- Ảnh `yellowvein_01`, `glare_01` là lá khỏe được thiết kế để gây báo động giả (failure cases).
