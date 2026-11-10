"""Plant Leaf Health Assessment System - Interactive Streamlit UI.

Hệ thống chẩn đoán mức độ bệnh và đánh giá sức khỏe lá cây tự động
dựa trên phân đoạn đa tầng (Otsu, K-means, Hình thái học và BFS).

Hỗ trợ 3 chế độ trực quan chuyên sâu:
    1. Pipeline Chẩn đoán Đa tầng (Otsu + K-means + Hình thái học & Đếm ổ bệnh)
    2. So sánh Thuật toán Otsu (From Scratch vs OpenCV + Biểu đồ Histogram & Phương sai sigma_B^2)
    3. So sánh Thuật toán K-means (From Scratch vs OpenCV)
"""

import io
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.data_io import matrix_to_csv_bytes
from src.postprocess import (
    closing_scratch,
    fill_holes_bfs_scratch,
    keep_largest_connected_component_scratch,
)
from src.benchmark import (
    compute_image_similarity_metrics,
    compute_segmentation_metrics_scratch,
    run_opencv_kmeans,
    run_opencv_otsu,
)
from src.pipeline import (
    build_pathology_feature_space,
    call_kmeans,
    call_load_image,
    call_otsu,
    call_preprocess,
    run_full_diagnosis_pipeline,
)

# =============================================================================
# CẤU HÌNH TRANG & CSS
# =============================================================================
st.set_page_config(
    page_title="Plant Leaf Health Assessment",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #374151;
        margin-bottom: 1.2rem;
    }
    .severity-badge {
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 600;
        margin: 10px 0;
        border-left: 6px solid;
    }
    .sev-0 { background-color: #ecfdf5; border-color: #10b981; color: #065f46; }
    .sev-1 { background-color: #f0fdf4; border-color: #84cc16; color: #3f6212; }
    .sev-2 { background-color: #fffbeb; border-color: #f59e0b; color: #92400e; }
    .sev-3 { background-color: #fff7ed; border-color: #f97316; color: #9a3412; }
    .sev-4 { background-color: #fef2f2; border-color: #ef4444; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HÀM TIỆN ÍCH NỘI BỘ GIAO DIỆN
# =============================================================================

_ST_MAJOR, _ST_MINOR = (int(x) for x in st.__version__.split(".")[:2])
_STRETCH_KW: Dict[str, Any] = (
    {"width": "stretch"} if (_ST_MAJOR, _ST_MINOR) >= (1, 50) else {"use_container_width": True}
)

def create_sample_leaf_image(preset: str = "halo") -> np.ndarray:
    """Tạo nhanh ảnh mẫu lá cây thực địa giả lập ngay trên bộ nhớ để chạy thử tức thì.

    Args:
        preset:
            - "halo": Mô phỏng Ảnh 1 (lá có các ổ bệnh gồm tâm nâu phân mảnh nằm trong quầng vàng lớn).
            - "field": Mô phỏng Ảnh 2 (lá bóng nắng có nhiều đốm bệnh nhỏ < 3% & nền đất/gỗ nâu góc trái).
    """
    H, W = 320, 320
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    if preset == "field":
        # Nền sáng ngoài trời kèm góc đất/gỗ nâu mờ ở góc trái
        img[:] = [230, 232, 228]
        soil_corner = (xx + 0.75 * yy) < 135
        img[soil_corner] = [142, 102, 65]
        wood_patch = (xx < 95) & (yy > 205)
        img[wood_patch] = [160, 122, 78]

        leaf_mask = ((xx - 175) / 92.0) ** 2 + ((yy - 160) / 122.0) ** 2 <= 1.0
        img[leaf_mask] = [38, 138, 68]

        # Vùng chói bóng nắng trên mặt lá (R, G, B cao nhưng G > R)
        glare = (((xx - 165) / 36.0) ** 2 + ((yy - 108) / 26.0) ** 2 <= 1.0) & leaf_mask
        img[glare] = [205, 228, 198]

        # 6 đốm bệnh nhỏ màu vàng-nâu (< 3% diện tích lá)
        small_spots = [
            (142, 132, 5), (195, 118, 4), (218, 162, 5),
            (158, 196, 5), (190, 212, 4), (132, 172, 4),
        ]
        for cx, cy, r in small_spots:
            sp = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
            img[sp & leaf_mask] = [168, 136, 42]
            core = ((xx - cx) ** 2 + (yy - cy) ** 2) <= max(1, r - 2) ** 2
            img[core & leaf_mask] = [115, 68, 28]
        return img

    # Preset "halo": Mô phỏng Ảnh 1 - nhiều chấm nâu nằm trong quầng vàng lớn
    img[:] = [238, 240, 238]
    leaf_mask = ((xx - 160) / 98.0) ** 2 + ((yy - 160) / 128.0) ** 2 <= 1.0
    img[leaf_mask] = [42, 142, 74]

    main_vein = (np.abs(xx - 160) <= 2) & leaf_mask
    img[main_vein] = [88, 185, 118]
    for offset_y in (-55, -15, 25, 65):
        side_vein = (np.abs((yy - (160 + offset_y)) - 0.35 * np.abs(xx - 160)) <= 1.5) & leaf_mask
        img[side_vein] = [78, 175, 108]

    # 3 ổ bệnh lớn: mỗi ổ có quầng vàng (Chlorosis) bao quanh nhiều chấm nâu (Necrosis) sát nhau
    halo_lesions = [
        (126, 115, 24, [(-6, -5, 4), (5, -2, 4), (-2, 6, 5), (7, 5, 3)]),
        (196, 178, 20, [(-5, -3, 4), (4, 3, 4), (0, -6, 3)]),
        (134, 214, 16, [(-3, -2, 4), (4, 2, 3)]),
    ]
    for cx, cy, hr, dots in halo_lesions:
        halo_outer = (((xx - cx) ** 2 + (yy - cy) ** 2) <= hr ** 2) & leaf_mask
        img[halo_outer] = [196, 198, 52]  # Quầng vàng tươi
        for dx, dy, dr in dots:
            dot = (((xx - (cx + dx)) ** 2 + (yy - (cy + dy)) ** 2) <= dr ** 2) & leaf_mask
            img[dot] = [102, 54, 22]  # Tâm nâu hoại tử

    # Nhiễu hạt nhỏ 2x2
    img[75:77, 155:157] = [145, 75, 25]
    return img


def image_to_png_bytes(img_array: np.ndarray) -> bytes:
    """Chuyển đổi mảng NumPy (2D hoặc 3D uint8) sang bytes PNG để tải xuống."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def safe_matrix_to_csv_bytes(matrix_2d: np.ndarray) -> bytes:
    """Chuyển đổi ma trận 2D sang bytes CSV (dùng `data_io` hoặc fallback nội bộ)."""
    try:
        return matrix_to_csv_bytes(matrix_2d)
    except NotImplementedError:
        buf = io.StringIO()
        np.savetxt(buf, matrix_2d, fmt="%d", delimiter=",")
        return buf.getvalue().encode("utf-8")


def plot_otsu_histogram_and_variance(
    histogram: np.ndarray,
    variance_curve: np.ndarray,
    threshold_scratch: int,
    threshold_opencv: int,
) -> plt.Figure:
    """Vẽ biểu đồ đôi: Phân bố Histogram mức xám và đường cong phương sai sigma_B^2(t)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    x = np.arange(256)

    # Biểu đồ Histogram
    ax1.bar(x, histogram, color="#2e7d32", alpha=0.75, width=1.0, label="Tần suất mức độ (Histogram)")
    ax1.axvline(threshold_scratch, color="#dc2626", linestyle="--", linewidth=2, label=f"Ngưỡng T* Scratch = {threshold_scratch}")
    ax1.axvline(threshold_opencv, color="#2563eb", linestyle=":", linewidth=2, label=f"Ngưỡng T* OpenCV = {threshold_opencv}")
    ax1.set_ylabel("Số lượng pixel")
    ax1.set_title("Biểu đồ Histogram & Vị trí Ngưỡng Cắt Otsu T*")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    # Biểu đồ Phương sai giữa các lớp sigma_B^2(t)
    ax2.plot(x, variance_curve, color="#d97706", linewidth=2, label=r"Phương sai giữa các lớp $\sigma_B^2(t)$")
    ax2.axvline(threshold_scratch, color="#dc2626", linestyle="--", linewidth=2)
    ax2.set_xlabel("Mức độ sáng (0..255)")
    ax2.set_ylabel(r"$\sigma_B^2(t)$")
    ax2.set_title(r"Đường cong Tối ưu hóa Phương sai Giữa các lớp $\sigma_B^2(t)$")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def build_metrics_dataframe(metrics: Dict[str, float], time_scratch_ms: float, time_cv_ms: float) -> pd.DataFrame:
    """Tạo bảng đối sánh định lượng chuẩn hóa cho Streamlit."""
    rows = [
        {"Chỉ số đánh giá": "IoU (Jaccard Index)", "Giá trị": f"{metrics['iou_pct']:.2f}%"},
        {"Chỉ số đánh giá": "Dice Score (F1)", "Giá trị": f"{metrics['dice_pct']:.2f}%"},
        {"Chỉ số đánh giá": "Pixel Accuracy", "Giá trị": f"{metrics['pixel_accuracy_pct']:.2f}%"},
        {"Chỉ số đánh giá": "Precision", "Giá trị": f"{metrics['precision_pct']:.2f}%"},
        {"Chỉ số đánh giá": "Recall", "Giá trị": f"{metrics['recall_pct']:.2f}%"},
        {"Chỉ số đánh giá": "MSE (Mean Squared Error)", "Giá trị": f"{metrics['mse']:.4f}"},
        {"Chỉ số đánh giá": "PSNR (dB)", "Giá trị": f"{metrics['psnr']:.2f} dB"},
        {"Chỉ số đánh giá": "Thời gian chạy Scratch", "Giá trị": f"{time_scratch_ms:.2f} ms"},
        {"Chỉ số đánh giá": "Thời gian chạy OpenCV", "Giá trị": f"{time_cv_ms:.2f} ms"},
    ]
    return pd.DataFrame(rows)


# =============================================================================
# GIAO DIỆN CHÍNH
# =============================================================================

st.markdown('<div class="main-header">🍃 Hệ Thống Chẩn Đoán Sức Khỏe Lá Cây Tự Động</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Phân đoạn đa tầng: <b>Otsu (Vegetation Map ExG×S + Lấp lỗ BFS) + K-Means Đặc trưng bệnh học + Hình thái học & BFS Đếm Ổ Bệnh</b> '
    '— Cài đặt 100% From Scratch bằng NumPy & Đối sánh định lượng OpenCV'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR: NGUỒN ẢNH, CHẾ ĐỘ & SIÊU THAM SỐ
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Bảng Điều Khiển")

app_mode = st.sidebar.selectbox(
    "Chọn chế độ vận hành:",
    [
        "1. Pipeline Chẩn đoán Đa tầng (Otsu + K-means + Hình thái học & Đếm ổ bệnh)",
        "2. So sánh Thuật toán Otsu (From Scratch vs OpenCV + Biểu đồ)",
        "3. So sánh Thuật toán K-means (From Scratch vs OpenCV)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Dữ liệu đầu vào")
input_source_type = st.sidebar.radio(
    "Chọn nguồn ảnh / ma trận:",
    [
        "Tạo nhanh ảnh mẫu lá cây có đốm bệnh (In-Memory)",
        "Tải lên file ảnh (.png, .jpg, .jpeg)",
        "Tải lên file ma trận điểm ảnh (.csv)",
    ],
)

raw_input_source: Any = None
if input_source_type == "Tạo nhanh ảnh mẫu lá cây có đốm bệnh (In-Memory)":
    sample_preset = st.sidebar.selectbox(
        "Chọn kịch bản ảnh mẫu:",
        [
            "Mẫu 1: Đốm tâm nâu & quầng vàng lớn (Chlorotic Halo)",
            "Mẫu 2: Lá bóng nắng, đốm nhỏ (<3%) & nền đất nâu góc ảnh",
        ],
    )
    preset_key = "halo" if sample_preset.startswith("Mẫu 1") else "field"
    raw_input_source = create_sample_leaf_image(preset=preset_key)
    st.sidebar.success(f"Đã tạo ảnh mẫu giả lập 320x320 ({preset_key}).")
elif input_source_type == "Tải lên file ảnh (.png, .jpg, .jpeg)":
    uploaded_img = st.sidebar.file_uploader("Chọn ảnh lá cây:", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        raw_input_source = uploaded_img
elif input_source_type == "Tải lên file ma trận điểm ảnh (.csv)":
    uploaded_csv = st.sidebar.file_uploader("Chọn file ma trận CSV:", type=["csv"])
    if uploaded_csv is not None:
        raw_input_source = uploaded_csv

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Tham số thuật toán")
blur_kernel = st.sidebar.select_slider("Kích thước lọc mịn Gauss (Kernel):", options=[3, 5, 7, 9, 11], value=5)
blur_sigma = st.sidebar.slider("Độ lệch chuẩn Gaussian (Sigma):", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
k_clusters = st.sidebar.slider("Số cụm K-Means (K):", min_value=2, max_value=6, value=4, step=1)
yellow_halo_sensitivity = st.sidebar.slider(
    "Độ nhạy bao phủ quầng vàng (Yellow Halo Sensitivity):",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
    help="Tăng để mở rộng vùng bắt quầng vàng xung quanh đốm bệnh và các đốm nhỏ; giảm để chỉ tập trung vào tâm hoại tử nâu.",
)
morph_kernel = st.sidebar.select_slider("Kích thước kernel Hình thái học (Closing):", options=[3, 5, 7, 9], value=5)
min_lesion_area = st.sidebar.slider("Diện tích ổ bệnh tối thiểu (min_area px):", min_value=5, max_value=200, value=20, step=5)
run_benchmark = st.sidebar.checkbox("Bật đối sánh định lượng với OpenCV", value=True)

if raw_input_source is None:
    st.info("👈 Vui lòng chọn ảnh mẫu hoặc tải lên file ảnh / ma trận CSV ở thanh bên trái để bắt đầu.")
    st.stop()


# =============================================================================
# CHẾ ĐỘ 1: PIPELINE CHẨN ĐOÁN ĐA TẦNG
# =============================================================================
if app_mode.startswith("1."):
    with st.spinner("Đang thực thi quy trình phân đoạn đa tầng từ đầu bằng NumPy..."):
        result = run_full_diagnosis_pipeline(
            file_source=raw_input_source,
            k_clusters=k_clusters,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            min_lesion_area=min_lesion_area,
            morph_kernel=morph_kernel,
            yellow_halo_sensitivity=yellow_halo_sensitivity,
            run_benchmark_comparison=run_benchmark,
        )

    # Hiển thị thông báo trạng thái fallback minh bạch
    if result["using_fallback"]:
        fb_list = ", ".join(result["fallback_modules"])
        st.info(
            f"ℹ️ **Chế độ chạy độc lập (Fallback Active):** Các module `{fb_list}` đang dùng cài đặt dự phòng nội bộ "
            "do nhánh của thành viên chưa merge. Các module của Nhóm trưởng (`postprocess`, `benchmark`, `pipeline`) đang chạy bản chính thức."
        )
    else:
        st.success("✅ **Full Native Mode:** Tất cả các module từ toàn bộ các nhánh thành viên đã được tích hợp và chạy trực tiếp!")

    # --- KẾT QUẢ CHẨN ĐOÁN Y BẠ CÂY TRỒNG (NỔI BẬT) ---
    diag = result["diagnosis"]
    sev_lvl = diag["severity_level"]
    st.subheader("🩺 Kết Quả Chẩn Đoán Sức Khỏe Lá Cây")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cấp độ bệnh", f"Cấp {sev_lvl} / 4")
    m2.metric("Tỷ lệ nhiễm bệnh", f"{diag['disease_ratio_percent']:.2f}%")
    m3.metric("Số ổ bệnh (BFS)", f"{diag['num_lesions']} ổ")
    m4.metric("Diện tích phiến lá", f"{diag['leaf_area_px']:,} px")
    m5.metric("Diện tích vùng bệnh", f"{diag['disease_area_px']:,} px")

    st.markdown(
        f'<div class="severity-badge sev-{sev_lvl}">'
        f'<b>Chẩn đoán: {diag["severity_label"]}</b><br/>'
        f'💡 <i>Khuyến nghị xử lý:</i> {diag["recommendation"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- HIỂN THỊ KẾT QUẢ ĐẦU CUỐI: ẢNH KHOANH VÙNG BBOX & BẢNG CHI TIẾT Ổ BỆNH ---
    st.markdown("---")
    st.subheader("🎯 Phát Hiện Ổ Bệnh & Hộp Bao (Bounding Boxes)")
    col_res1, col_res2 = st.columns([1.15, 1])
    with col_res1:
        st.image(
            result["annotated_img"],
            caption=f"Ảnh khoanh vùng {diag['num_lesions']} ổ bệnh độc lập (BFS 8-hướng, sắp xếp diện tích giảm dần)",
            **_STRETCH_KW,
        )
    with col_res2:
        bboxes = result["postprocess_result"]["bboxes"]
        if bboxes:
            bbox_df = pd.DataFrame(
                [
                    {
                        "Ổ bệnh": f"#{i}",
                        "Tọa độ (x, y)": f"({b[0]}, {b[1]})",
                        "Kích thước (w x h)": f"{b[2]} x {b[3]}",
                        "Diện tích (px)": b[4],
                    }
                    for i, b in enumerate(bboxes, start=1)
                ]
            )
            st.markdown("**Danh sách chi tiết các ổ bệnh (sắp xếp theo diện tích giảm dần):**")
            st.dataframe(bbox_df, **_STRETCH_KW, height=280)
        else:
            st.success("Không phát hiện ổ bệnh nào có diện tích >= ngưỡng min_area.")

    # --- TRỰC QUAN CÁC BƯỚC TRUNG GIAN ---
    st.markdown("---")
    st.subheader("🔬 Trực Quan Hóa Các Tầng Xử Lý Trung Gian")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.image(result["original_rgb"], caption=f"1. Ảnh gốc RGB {result['original_rgb'].shape}", **_STRETCH_KW)
    with s2:
        st.image(result["otsu_channel"], caption="2. Bản đồ Thực vật (ExG × Saturation)", **_STRETCH_KW)
    with s3:
        st.image(
            result["leaf_mask"],
            caption=f"3. Leaf ROI Otsu (T* = {result['otsu_result']['threshold']}) + Lấp lỗ BFS",
            **_STRETCH_KW,
        )

    s4, s5, s6 = st.columns(3)
    with s4:
        st.image(
            result["kmeans_result"]["segmented_img"],
            caption=f"4. Phân cụm K-Means Đặc trưng bệnh học (k={k_clusters}, {result['kmeans_result']['n_iter']} vòng lặp)",
            **_STRETCH_KW,
        )
    with s5:
        st.image(
            result["color_result"]["disease_mask"],
            caption=f"5. Mặt nạ Bệnh thô (Tâm nâu + Quầng vàng, nhạy={yellow_halo_sensitivity:.2f})",
            **_STRETCH_KW,
        )
    with s6:
        st.image(
            result["postprocess_result"]["cleaned_mask"],
            caption=f"6. Mặt nạ Bệnh sau Closing (k={morph_kernel}) -> Opening & Lọc BFS",
            **_STRETCH_KW,
        )

    # --- ĐỐI SÁNH ĐỊNH LƯỢNG OPENCV & TẢI XUẤT KẾT QUẢ ---
    if run_benchmark and result["benchmark"]:
        st.markdown("---")
        st.subheader("📊 Đối Sánh Định Lượng (Scratch vs OpenCV)")
        bm = result["benchmark"]
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("**Tầng 1: Tách nền Otsu (Scratch vs `cv2.threshold`)**")
            df_otsu = build_metrics_dataframe(
                bm["otsu_metrics"],
                result["otsu_result"]["exec_time_ms"],
                bm["otsu_cv"]["exec_time_ms"],
            )
            st.table(df_otsu)
        with b_col2:
            st.markdown("**Tầng 2: Tách bệnh K-Means (Scratch vs `cv2.kmeans`)**")
            df_km = build_metrics_dataframe(
                bm["kmeans_disease_metrics"],
                result["kmeans_result"]["exec_time_ms"],
                bm["kmeans_cv"]["exec_time_ms"],
            )
            st.table(df_km)

    st.markdown("---")
    st.subheader("💾 Xuất & Tải Xuống Kết Quả")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="📥 Tải ảnh khoanh vùng ổ bệnh (.png)",
            data=image_to_png_bytes(result["annotated_img"]),
            file_name="leaf_lesions_detected.png",
            mime="image/png",
            **_STRETCH_KW,
        )
    with dl2:
        st.download_button(
            label="📥 Tải mặt nạ vùng bệnh (.png)",
            data=image_to_png_bytes(result["postprocess_result"]["cleaned_mask"]),
            file_name="disease_cleaned_mask.png",
            mime="image/png",
            **_STRETCH_KW,
        )
    with dl3:
        st.download_button(
            label="📥 Tải ma trận mặt nạ bệnh (.csv)",
            data=safe_matrix_to_csv_bytes(result["postprocess_result"]["cleaned_mask"]),
            file_name="disease_mask_matrix.csv",
            mime="text/csv",
            **_STRETCH_KW,
        )


# =============================================================================
# CHẾ ĐỘ 2: SO SÁNH THUẬT TOÁN OTSU (FROM SCRATCH VS OPENCV)
# =============================================================================
elif app_mode.startswith("2."):
    st.subheader("⚖️ So Sánh Chuyên Sâu Thuật Toán Otsu: NumPy Scratch vs OpenCV")
    fb_tracker: List[str] = []
    loaded = call_load_image(raw_input_source, max_size=512, fallback_tracker=fb_tracker)
    prep = call_preprocess(loaded, blur_kernel=blur_kernel, blur_sigma=blur_sigma, fallback_tracker=fb_tracker)

    channel_choice = st.radio(
        "Chọn kênh ảnh đưa vào thuật toán Otsu:",
        [
            "Bản đồ Thực vật Vegetation Map (ExG × S) — Chống nhiễu nền đất nâu",
            "Kênh Độ bão hòa Saturation (HSV-S)",
            "Kênh Mức xám Grayscale",
        ],
        horizontal=True,
    )
    if channel_choice.startswith("Bản đồ"):
        target_channel = prep["vegetation_map"]
    elif channel_choice.startswith("Kênh Độ"):
        target_channel = prep["saturation"]
    else:
        target_channel = prep["grayscale"]

    otsu_scratch_res = call_otsu(target_channel, fallback_tracker=fb_tracker)
    otsu_cv_res = run_opencv_otsu(target_channel)
    otsu_metrics = compute_segmentation_metrics_scratch(
        otsu_scratch_res["binary_mask"], otsu_cv_res["binary_mask"]
    )

    c1, c2, c3 = st.columns(3)
    c1.image(target_channel, caption="Ảnh đầu vào của Otsu (sau lọc Gauss)", **_STRETCH_KW)
    c2.image(
        otsu_scratch_res["binary_mask"],
        caption=f"Otsu Scratch (T* = {otsu_scratch_res['threshold']}, {otsu_scratch_res['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )
    c3.image(
        otsu_cv_res["binary_mask"],
        caption=f"Otsu OpenCV (T* = {otsu_cv_res['threshold']}, {otsu_cv_res['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )

    st.markdown("---")
    chart_col, metric_col = st.columns([1.4, 1])
    with chart_col:
        fig = plot_otsu_histogram_and_variance(
            otsu_scratch_res["histogram"],
            otsu_scratch_res["variance_curve"],
            otsu_scratch_res["threshold"],
            otsu_cv_res["threshold"],
        )
        st.pyplot(fig)
        plt.close(fig)
    with metric_col:
        st.markdown("**Bảng chỉ số đối sánh Otsu (Scratch vs OpenCV):**")
        st.table(
            build_metrics_dataframe(
                otsu_metrics,
                otsu_scratch_res["exec_time_ms"],
                otsu_cv_res["exec_time_ms"],
            )
        )


# =============================================================================
# CHẾ ĐỘ 3: SO SÁNH THUẬT TOÁN K-MEANS (FROM SCRATCH VS OPENCV)
# =============================================================================
else:
    st.subheader("🎨 So Sánh Thuật Toán Phân Cụm K-Means: NumPy Vectorized Scratch vs OpenCV")
    fb_tracker = []
    loaded = call_load_image(raw_input_source, max_size=512, fallback_tracker=fb_tracker)
    prep = call_preprocess(loaded, blur_kernel=blur_kernel, blur_sigma=blur_sigma, fallback_tracker=fb_tracker)

    use_leaf_roi = st.checkbox("Chỉ phân cụm trong vùng phiến lá (Loại nền nâu + Lấp lỗ BFS)", value=True)
    use_pathology_space = st.checkbox("Dùng không gian đặc trưng nhạy bệnh học (RGB + R-G + NormRG + Hue)", value=True)
    roi_mask = None
    if use_leaf_roi:
        otsu_res = call_otsu(prep["otsu_input"], fallback_tracker=fb_tracker)
        H, W = prep["grayscale"].shape
        roi_close_k = 11 if min(H, W) >= 128 else 7
        roi_mask = fill_holes_bfs_scratch(
            keep_largest_connected_component_scratch(
                closing_scratch(otsu_res["binary_mask"], kernel_size=roi_close_k)
            )
        )

    km_input = (
        build_pathology_feature_space(prep["blurred_rgb"], prep["hsv"])
        if use_pathology_space
        else prep["blurred_rgb"]
    )
    km_scratch = call_kmeans(
        km_input, k=k_clusters, max_iter=100, tol=1e-3, random_state=42, roi_mask=roi_mask, fallback_tracker=fb_tracker
    )
    km_cv = run_opencv_kmeans(
        km_input, k=k_clusters, max_iter=100, tol=1e-3, roi_mask=roi_mask
    )
    seg_scratch_rgb = km_scratch["segmented_img"][:, :, :3].astype(np.uint8)
    seg_cv_rgb = km_cv["segmented_img"][:, :, :3].astype(np.uint8)
    centroids_rgb = km_scratch["centroids"][:, :3]
    sim_metrics = compute_image_similarity_metrics(seg_scratch_rgb, seg_cv_rgb)

    k1, k2, k3 = st.columns(3)
    k1.image(prep["blurred_rgb"], caption="Ảnh RGB sau làm mịn Gauss", **_STRETCH_KW)
    k2.image(
        seg_scratch_rgb,
        caption=f"K-Means Scratch (k={k_clusters}, {km_scratch['n_iter']} iter, {km_scratch['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )
    k3.image(
        seg_cv_rgb,
        caption=f"K-Means OpenCV (k={k_clusters}, {km_cv['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )

    st.markdown("---")
    km_col1, km_col2 = st.columns(2)
    with km_col1:
        st.markdown("**Tọa độ tâm cụm RGB (Scratch Centroids):**")
        cent_df = pd.DataFrame(
            np.round(centroids_rgb, 1),
            columns=["R", "G", "B"],
        )
        cent_df.index.name = "Cluster ID"
        st.dataframe(cent_df, **_STRETCH_KW)
    with km_col2:
        st.markdown("**Chỉ số tương đồng ảnh phân đoạn (Scratch vs OpenCV):**")
        sim_df = pd.DataFrame(
            [
                {"Chỉ số": "MSE (Mean Squared Error)", "Giá trị": f"{sim_metrics['mse']:.4f}"},
                {"Chỉ số": "PSNR (dB)", "Giá trị": f"{sim_metrics['psnr']:.2f} dB"},
                {"Chỉ số": "Thời gian chạy Scratch", "Giá trị": f"{km_scratch['exec_time_ms']:.2f} ms"},
                {"Chỉ số": "Thời gian chạy OpenCV", "Giá trị": f"{km_cv['exec_time_ms']:.2f} ms"},
                {"Chỉ số": "Số vòng lặp hội tụ (Scratch)", "Giá trị": f"{km_scratch['n_iter']}"},
            ]
        )
        st.table(sim_df)
