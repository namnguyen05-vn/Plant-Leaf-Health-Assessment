# Kết quả thực nghiệm Vai trò 2 — Histogram & Otsu

Tự sinh bởi `python -m experiments.otsu_analysis`. Dataset: `data/raw/` (30 ảnh có Ground Truth).

## 1. Độ lệch ngưỡng T* tự code so với OpenCV

- Số lần so sánh: **150** (30 ảnh × 4 kênh tách nền + 30 lần Otsu trong ROI lá)
- |ΔT| lớn nhất: **0**, trung bình: 0.000, số lần trùng khớp tuyệt đối: 150/150
- Yêu cầu |ΔT| ≤ 1: **ĐẠT**

## 2. Tách nền lá (Tầng 1) — IoU trung bình so với GT

Sau Otsu chỉ giữ thành phần liên thông lớn nhất (LCC). Cột `raw` là mặt nạ Otsu thô.

| Kênh | IoU raw | IoU LCC | Dice LCC | IoU LCC thấp nhất (ảnh) |
|---|---|---|---|---|
| Gray | 0.585 | 0.589 | 0.657 | 0.000 (mild_01.jpg) |
| S (HSV) | 0.881 | 0.902 | 0.921 | 0.000 (moderate_05.jpg) |
| ExG | 0.863 | 0.830 | 0.890 | 0.119 (yellowvein_01.jpg) |
| VegMap (pipeline) | 0.944 | 0.942 | 0.968 | 0.646 (critical_02.jpg) |

IoU LCC theo nền (kênh tốt nhất mỗi nền in đậm):

| Nền | Gray | S (HSV) | ExG | VegMap (pipeline) |
|---|---|---|---|---|
| dark | 0.967 | 0.956 | 0.929 | **0.970** |
| gray | 0.689 | **0.972** | 0.731 | 0.970 |
| paper | 0.864 | **0.984** | 0.849 | 0.976 |
| soil | 0.502 | **0.943** | 0.848 | 0.926 |
| wood | 0.138 | 0.701 | 0.825 | **0.888** |

IoU LCC theo điều kiện chụp:

| Điều kiện | Gray | S (HSV) | ExG | VegMap (pipeline) |
|---|---|---|---|---|
| normal | 0.616 | 0.884 | 0.843 | 0.934 |
| shadow | 0.206 | 0.983 | 0.891 | 0.956 |
| glare | 0.887 | 0.879 | 0.867 | 0.957 |
| noise | 0.492 | 0.965 | 0.894 | 0.968 |
| yellow_vein | 0.481 | 0.994 | 0.119 | 0.977 |
| lowlight | 0.455 | 0.980 | 0.938 | 0.958 |

## 3. Otsu phân ngưỡng đốm bệnh trong ROI lá (kênh Hue)

| Nhóm | IoU TB | R% GT TB | R% Otsu TB |
|---|---|---|---|
| Khỏe | — (không có vết bệnh) | 0.00 | 32.52 |
| Nhẹ | 0.783 | 5.03 | 4.19 |
| Trung bình | 0.794 | 17.65 | 14.47 |
| Nặng | 0.829 | 41.41 | 34.57 |
| Nghiêm trọng | 0.856 | 66.47 | 56.98 |
| Điều kiện khó | 0.830 | 8.14 | 12.24 |

## 4. Độ chồng lấn histogram giữa các lớp pixel (càng nhỏ càng dễ phân ngưỡng)

| Kênh | Nền ↔ Lá khỏe | Lá khỏe ↔ Vết bệnh |
|---|---|---|
| R | 0.101 | 0.223 |
| G | 0.341 | 0.397 |
| B | 0.330 | 0.455 |
| Gray | 0.429 | 0.303 |
| H | 0.031 | 0.047 |
| S | 0.130 | 0.816 |
| V | 0.611 | 0.615 |
| ExG | 0.004 | 0.090 |

## 5. Tốc độ (ảnh 512×512, trung vị 5 lần chạy)

| Phép toán | Tự code (ms) | OpenCV (ms) | Chậm hơn (lần) | Sai khác pixel lớn nhất |
|---|---|---|---|---|
| RGB→Gray | 2.199 | 0.064 | 34.2× | 1 |
| RGB→HSV | 41.105 | 0.234 | 175.3× |  |
| Gaussian 5x5 (RGB) | 15.832 | 0.402 | 39.4× | 1 |
| Mean 5x5 (Gray) | 5.331 | 0.105 | 51.0× | 0 |
| Otsu (Gray) | 0.873 | 0.187 | 4.7× | 0 |
