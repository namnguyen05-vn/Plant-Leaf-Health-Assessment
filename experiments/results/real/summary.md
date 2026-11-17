# Kết quả pipeline trên ảnh lá thật (`data/raw/real/`)

Tự sinh bởi `python -m experiments.run_real_images`. Ảnh thật không có Ground Truth: cột *ước lượng* là đánh giá bằng mắt, chỉ mang tính tham khảo.

| Nhóm | Số ảnh | Khớp | Thấp hơn | Cao hơn |
|---|---|---|---|---|
| Lá khỏe (tự chụp) | 2 | 2 | 0 | 0 |
| Lá bệnh | 8 | 7 | 1 | 0 |
| Ca khó | 4 | 2 | 2 | 0 |
| Giới hạn phương pháp | 3 | 1 | 1 | 0 |

| Ảnh | Nhóm | Ước lượng (cấp / R%) | Dự đoán (cấp / R%) | Kết quả | Lá/khung |
|---|---|---|---|---|---|
| cab_lv0_01_selfshot.jpg | healthy | 0 / 0-0 | 0 / 0.0 | khớp | 0.336 |
| cab_lv0_02_selfshot.jpg | healthy | 0 / 0-1 | 0 / 0.97 | khớp | 0.366 |
| cab_lv1_01_bugwood1572855.jpg | diseased | 1 / 5-8 | 1 / 3.5 | khớp | 0.586 |
| cab_lv1_02_bugwood1572869.jpg | diseased | 1 / 3-6 | 1 / 7.93 | khớp | 0.334 |
| cab_lv1_03_bugwood1634015.jpg | diseased | 1 / 3-5 | 1 / 6.78 | khớp | 0.185 |
| cab_lv1_04_bugwood1572866_fullframe.jpg | diseased | 1 / 2-4 | 1 / 3.84 | khớp | 0.046 |
| cab_lv1_05_bugwood1572864_fullframe.jpg | diseased | 1-2 / 8-12 | 1 / 3.15 | khớp | 0.492 |
| cab_lv2_01_internet_blackbg.jpg | diseased | 1-2 / 8-12 | 1 / 6.82 | khớp | 0.721 |
| cab_lv2_02_bugwood5403519_soil_shadow.jpg | hard | 2 / 10-15 | 2 / 12.95 | khớp | 0.66 |
| cab_lv2_03_internet_field.jpg | hard | 2 / 15-20 | 2 / 16.68 | khớp | 0.382 |
| cab_lv3_01_bugwood1534036.jpg | diseased | 2-3 / 25-35 | 2 / 20.43 | khớp | 0.667 |
| cab_lv3_02_internet_blackbg.jpg | diseased | 2-3 / 20-30 | 1 / 6.81 | thấp hơn | 0.672 |
| cab_lv3_03_internet_wood.jpg | hard | 3 / 40-50 | 1 / 4.32 | thấp hơn | 0.42 |
| cab_lv4_01_internet_field.jpg | hard | 3-4 / 50-70 | 0 / 0.0 | thấp hơn | 0.763 |
| lim_01_bugwood5389805_purple_spots.jpg | limitation | 2 / 15-25 | 1 / 1.83 | thấp hơn | 0.595 |
| lim_02_bugwood1563086_cucumber_white_spots.jpg | limitation | 1 / 2-3 | 1 / 2.56 | khớp | 0.357 |
| lim_03_internet_cucumber_mosaic.jpg | limitation | ? / ? | 0 / 0.15 | n/a | 0.558 |
