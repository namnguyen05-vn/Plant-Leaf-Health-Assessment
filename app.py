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
from src.benchmark import (
    compute_image_similarity_metrics,
    compute_segmentation_metrics_scratch,
    run_opencv_kmeans,
    run_opencv_otsu,
)
from src.pipeline import (
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

def create_sample_leaf_image() -> np.ndarray:
    """Tạo nhanh ảnh mẫu lá cây có đốm bệnh ngay trên bộ nhớ để chạy thử tức thì."""
    H, W = 320, 320
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    # Nền sáng trung tính
    img[:] = [238, 240, 238]

    # Phiến lá hình elip nghiêng
    leaf_mask = ((xx - 160) / 96.0) ** 2 + ((yy - 160) / 128.0) ** 2 <= 1.0
    img[leaf_mask] = [42, 142, 74]

    # Gân lá chính và gân phụ
    main_vein = (np.abs(xx - 160) <= 2) & leaf_mask
    img[main_vein] = [88, 185, 118]
    for offset_y in (-55, -15, 25, 65):
        side_vein = (np.abs((yy - (160 + offset_y)) - 0.35 * np.abs(xx - 160)) <= 1.5) & leaf_mask
        img[side_vein] = [78, 175, 108]

    # Các đốm bệnh hoại tử với sắc nâu / vàng úa / đen
    spots = [
        (132, 118, 20, [139, 69, 19]),   # Ổ bệnh nâu sẫm lớn
        (195, 185, 15, [184, 120, 32]),  # Ổ bệnh vàng nâu
        (125, 205, 11, [160, 82, 45]),   # Ổ bệnh vừa
        (182, 115, 8, [120, 55, 22]),    # Ổ bệnh nhỏ
        (158, 232, 6, [150, 95, 25]),    # Ổ bệnh nhỏ
    ]
    for cx, cy, r, color in spots:
        spot_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        img[spot_mask & leaf_mask] = color

    # Nhiễu hạt nhỏ (sẽ bị lọc bỏ bởi Morphology + min_area)
    img[75:77, 155:157] = [145, 75, 25]
    img[225:227, 175:177] = [145, 75, 25]
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
    'Phân đoạn đa tầng: <b>Otsu (Kênh Saturation) + K-Means Vectorized + Hình thái học & BFS Đếm Ổ Bệnh</b> '
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
    raw_input_source = create_sample_leaf_image()
    st.sidebar.success("Đã tạo mẫu ảnh lá cây giả lập (320x320 RGB) chứa 5 ổ bệnh & nhiễu hạt.")
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
k_clusters = st.sidebar.slider("Số cụm K-Means (K):", min_value=2, max_value=6, value=3, step=1)
morph_kernel = st.sidebar.select_slider("Kích thước kernel Hình thái học:", options=[3, 5, 7], value=3)
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
        st.image(result["otsu_channel"], caption="2. Kênh Saturation (hoặc Gray) sau Gauss Blur", **_STRETCH_KW)
    with s3:
        st.image(
            result["leaf_mask"],
            caption=f"3. Mặt nạ Phiến lá Otsu (T* = {result['otsu_result']['threshold']})",
            **_STRETCH_KW,
        )

    s4, s5, s6 = st.columns(3)
    with s4:
        st.image(
            result["kmeans_result"]["segmented_img"],
            caption=f"4. Phân cụm K-Means (k={k_clusters}, {result['kmeans_result']['n_iter']} vòng lặp)",
            **_STRETCH_KW,
        )
    with s5:
        st.image(result["color_result"]["disease_mask"], caption="5. Mặt nạ Bệnh thô (Color Clustering)", **_STRETCH_KW)
    with s6:
        st.image(
            result["postprocess_result"]["cleaned_mask"],
            caption=f"6. Mặt nạ Bệnh sau Morphology & Lọc BFS (k={morph_kernel})",
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
        ["Kênh Độ bão hòa Saturation (HSV-S) — Tối ưu tách lá khỏi nền", "Kênh Mức xám Grayscale"],
        horizontal=True,
    )
    target_channel = prep["saturation"] if channel_choice.startswith("Kênh Độ") else prep["grayscale"]

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

    use_leaf_roi = st.checkbox("Chỉ phân cụm trong vùng phiến lá (Loại bỏ nền qua Otsu trước)", value=True)
    roi_mask = None
    if use_leaf_roi:
        otsu_res = call_otsu(prep["otsu_input"], fallback_tracker=fb_tracker)
        roi_mask = otsu_res["binary_mask"]

    km_scratch = call_kmeans(
        prep["blurred_rgb"], k=k_clusters, max_iter=100, tol=1e-3, random_state=42, roi_mask=roi_mask, fallback_tracker=fb_tracker
    )
    km_cv = run_opencv_kmeans(
        prep["blurred_rgb"], k=k_clusters, max_iter=100, tol=1e-3, roi_mask=roi_mask
    )
    sim_metrics = compute_image_similarity_metrics(km_scratch["segmented_img"], km_cv["segmented_img"])

    k1, k2, k3 = st.columns(3)
    k1.image(prep["blurred_rgb"], caption="Ảnh RGB sau làm mịn Gauss", **_STRETCH_KW)
    k2.image(
        km_scratch["segmented_img"],
        caption=f"K-Means Scratch (k={k_clusters}, {km_scratch['n_iter']} iter, {km_scratch['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )
    k3.image(
        km_cv["segmented_img"],
        caption=f"K-Means OpenCV (k={k_clusters}, {km_cv['exec_time_ms']:.2f} ms)",
        **_STRETCH_KW,
    )

    st.markdown("---")
    km_col1, km_col2 = st.columns(2)
    with km_col1:
        st.markdown("**Tọa độ tâm cụm RGB (Scratch Centroids):**")
        cent_df = pd.DataFrame(
            np.round(km_scratch["centroids"], 1),
            columns=[f"Kênh {i}" for i in range(km_scratch["centroids"].shape[1])],
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
