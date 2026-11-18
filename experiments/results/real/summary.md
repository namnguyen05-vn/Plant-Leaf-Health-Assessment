# Kết quả pipeline trên ảnh lá thật (`data/raw/real/`)

Tự sinh bởi `python -m experiments.run_real_images`. Ảnh thật không có Ground Truth: cột *ước lượng* là đánh giá bằng mắt, chỉ mang tính tham khảo.

| Nhóm | Số ảnh | Khớp | Thấp hơn | Cao hơn |
|---|---|---|---|---|
| Lá khỏe | 3 | 2 | 0 | 1 |
| Vết bệnh nâu/vàng rõ | 6 | 5 | 1 | 0 |
| Vàng úa lan rộng | 2 | 0 | 2 | 0 |
| Khảm/úa loang (+ bóng đổ) | 7 | 5 | 2 | 0 |
| Failure case | 4 | 1 | 3 | 0 |

| Ảnh | Nhóm | Ước lượng (cấp / R%) | Dự đoán (cấp / R%) | Kết quả | Lá/khung |
|---|---|---|---|---|---|
| fail_01_wood_deadtissue.jpg | failure | 3 / 40-50 | 1 / 4.32 | thấp hơn | 0.42 |
| fail_02_field_yellowed_lv4.jpg | failure | 3-4 / 50-70 | 0 / 0.0 | thấp hơn | 0.763 |
| fail_03_bugwood5389805_purple_spots.jpg | failure | 2 / 15-25 | 1 / 1.83 | thấp hơn | 0.595 |
| fail_04_bugwood1572866_fullframe.jpg | failure | 1 / 2-4 | 1 / 3.84 | khớp | 0.046 |
| tom_healthy_01.png | healthy | 0 / 0-1 | 1 / 1.68 | cao hơn | 0.424 |
| tom_healthy_02.png | healthy | 0 / 0-0 | 0 / 0.09 | khớp | 0.533 |
| tom_healthy_03.png | healthy | 0 / 0-1 | 0 / 0.07 | khớp | 0.439 |
| tom_lesion_01.png | lesion | 0-1 / 0.5-3 | 0 / 0.13 | khớp | 0.472 |
| tom_lesion_02.png | lesion | 1 / 3-8 | 0 / 0.29 | thấp hơn | 0.508 |
| tom_lesion_03.png | lesion | 1-2 / 8-15 | 1 / 5.8 | khớp | 0.52 |
| tom_lesion_04.png | lesion | 1-2 / 8-15 | 1 / 5.67 | khớp | 0.501 |
| tom_lesion_05.png | lesion | 1-2 / 8-15 | 1 / 6.28 | khớp | 0.481 |
| tom_lesion_06.png | lesion | 2-3 / 25-40 | 2 / 23.73 | khớp | 0.379 |
| tom_mottle_shadow_01.png | mottle_shadow | 1-3 / 10-40 | 1 / 3.35 | khớp | 0.26 |
| tom_mottle_shadow_02.png | mottle_shadow | 1-3 / 10-40 | 1 / 1.43 | khớp | 0.292 |
| tom_mottle_shadow_03.png | mottle_shadow | 1-3 / 10-40 | 1 / 3.8 | khớp | 0.309 |
| tom_mottle_shadow_04.png | mottle_shadow | 1-3 / 10-40 | 0 / 0.78 | thấp hơn | 0.304 |
| tom_mottle_shadow_05.png | mottle_shadow | 1-3 / 10-40 | 1 / 1.33 | khớp | 0.31 |
| tom_mottle_shadow_06.png | mottle_shadow | 1-3 / 10-40 | 1 / 1.76 | khớp | 0.288 |
| tom_mottle_shadow_07.png | mottle_shadow | 1-3 / 10-40 | 0 / 0.22 | thấp hơn | 0.198 |
| tom_yellow_01.png | yellowing | 2-3 / 20-35 | 1 / 4.0 | thấp hơn | 0.427 |
| tom_yellow_02.png | yellowing | 2-3 / 20-35 | 1 / 3.47 | thấp hơn | 0.34 |
