"""Plant Leaf Health Assessment System - Interactive Streamlit UI.

Hệ thống chẩn đoán mức độ bệnh và đánh giá sức khỏe lá cây tự động
dựa trên phân đoạn đa tầng (Otsu, K-means và Hình thái học).

Bộ khung giao diện Streamlit kết nối trực tiếp với `src/pipeline.py`.
Hệ thống xử lý ngoại lệ `NotImplementedError` một cách thân thiện để khi chạy
`streamlit run app.py` ngay từ lúc mới khởi tạo khung vẫn hiển thị giao diện mượt mà,
đồng thời thông báo trạng thái rõ ràng của từng module tính năng.
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Plant Leaf Health Assessment",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện trực quan và chuyên nghiệp
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6354;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #4caf50;
        margin-bottom: 10px;
    }
    .skeleton-notice {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_sample_leaf_image() -> np.ndarray:
    """Tạo ảnh lá cây giả lập 256x256 phục vụ demo giao diện khi chưa tải ảnh ngoài."""
    H, W = 256, 256
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    # Nền xám nhạt (240, 240, 240)
    img[:] = [240, 242, 240]

    # Phiến lá hình elip nghiêng (tâm 128, 128)
    leaf_mask = ((xx - 128) / 75) ** 2 + ((yy - 128) / 100) ** 2 <= 1.0
    img[leaf_mask] = [46, 139, 87]  # SeaGreen

    # Gân lá chính (nhạt hơn)
    vein_main = (np.abs(xx - 128) <= 2) & leaf_mask
    img[vein_main] = [102, 205, 170]

    # Vết bệnh hoại tử 1 (đốm nâu tâm 110, 100, bán kính 16)
    spot1 = ((xx - 110) ** 2 + (yy - 100) ** 2) <= 16 ** 2
    img[spot1 & leaf_mask] = [139, 69, 19]

    # Vết bệnh hoại tử 2 (đốm vàng nâu tâm 150, 160, bán kính 12)
    spot2 = ((xx - 150) ** 2 + (yy - 160) ** 2) <= 12 ** 2
    img[spot2 & leaf_mask] = [205, 133, 63]

    # Vết bệnh hoại tử 3 (đốm nhỏ tâm 100, 170, bán kính 8)
    spot3 = ((xx - 100) ** 2 + (yy - 170) ** 2) <= 8 ** 2
    img[spot3 & leaf_mask] = [160, 82, 45]

    return img


def draw_bounding_boxes(
    img_rgb: np.ndarray,
    bboxes: list,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ hộp bao BBox lên ảnh RGB bằng NumPy slicing đơn giản."""
    canvas = img_rgb.copy()
    H, W = canvas.shape[:2]
    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = max(0, min(W - 1, x))
        y1 = max(0, min(H - 1, y))
        x2 = max(0, min(W - 1, x + w))
        y2 = max(0, min(H - 1, y + h))
        canvas[y1 : min(y1 + thickness, H), x1 : x2 + 1] = color
        canvas[max(0, y2 - thickness) : y2 + 1, x1 : x2 + 1] = color
        canvas[y1 : y2 + 1, x1 : min(x1 + thickness, W)] = color
        canvas[y1 : y2 + 1, max(0, x2 - thickness) : x2 + 1] = color
    return canvas


# =============================================================================
# GIAO DIỆN CHÍNH (MAIN UI)
# =============================================================================

st.markdown('<div class="main-header">🍃 Plant Leaf Health Assessment System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Hệ thống chẩn đoán mức độ bệnh và đánh giá sức khỏe lá cây tự động dựa trên phân đoạn đa tầng '
    '<b>(Otsu Scratch + K-means Vectorized + Hình thái học & BFS)</b>'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR: Cấu hình tham số & Tải ảnh
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình hệ thống")

input_mode = st.sidebar.radio(
    "Nguồn ảnh đầu vào:",
    ["Ảnh Demo Mẫu (Synthetic Leaf)", "Tải ảnh từ máy tính (.jpg, .png)", "Tải file ma trận điểm ảnh (.csv)"],
)

input_image: Optional[np.ndarray] = None

if input_mode == "Ảnh Demo Mẫu (Synthetic Leaf)":
    input_image = create_sample_leaf_image()
    st.sidebar.success("Đang sử dụng ảnh mẫu giả lập có sẵn.")
elif input_mode == "Tải ảnh từ máy tính (.jpg, .png)":
    uploaded_file = st.sidebar.file_uploader("Chọn file ảnh lá cây:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            from src.data_io import load_image_or_csv

            input_image = load_image_or_csv(uploaded_file)
        except NotImplementedError:
            st.sidebar.warning("⚠️ `data_io.load_image_or_csv` đang phát triển. Tạm thời dùng PIL để đọc ảnh hiển thị.")
            from PIL import Image

            pil_img = Image.open(uploaded_file).convert("RGB")
            input_image = np.array(pil_img)
elif input_mode == "Tải file ma trận điểm ảnh (.csv)":
    uploaded_csv = st.sidebar.file_uploader("Chọn file ma trận CSV:", type=["csv"])
    if uploaded_csv is not None:
        try:
            from src.data_io import load_image_or_csv

            input_image = load_image_or_csv(uploaded_csv)
        except NotImplementedError:
            st.sidebar.warning("⚠️ `data_io` đang phát triển. Đọc thử nghiệm qua pandas.")
            df = pd.read_csv(uploaded_csv, header=None)
            input_image = df.to_numpy(dtype=np.uint8)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Siêu tham số thuật toán")
k_clusters = st.sidebar.slider("Số cụm K-Means (K):", min_value=2, max_value=6, value=3, step=1)
blur_kernel = st.sidebar.slider("Kernel làm mịn Gauss:", min_value=3, max_value=11, value=5, step=2)
blur_sigma = st.sidebar.slider("Gaussian Sigma:", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
morph_kernel = st.sidebar.slider("Kernel hình thái học (Opening/Closing):", min_value=3, max_value=9, value=3, step=2)
min_lesion_area = st.sidebar.slider("Ngưỡng diện tích ổ bệnh tối thiểu (min_area px):", min_value=5, max_value=100, value=20, step=5)
run_benchmark = st.sidebar.checkbox("Chạy đối sánh định lượng với OpenCV", value=True)

# -----------------------------------------------------------------------------
# TRẠNG THÁI KHỞI TẠO BỘ KHUNG (SKELETON STATUS DASHBOARD)
# -----------------------------------------------------------------------------
with st.expander("📌 Bảng trạng thái triển khai Module dự án (Skeleton Roadmap)", expanded=False):
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.info("**Module 1: I/O & Preprocess**\n- `src/data_io.py`\n- `src/preprocess.py`")
    with col_stat2:
        st.info("**Module 2: Phân đoạn tầng 1 & 2**\n- `src/otsu.py`\n- `src/kmeans.py`")
    with col_stat3:
        st.info("**Module 3: Màu sắc & Hình thái**\n- `src/color_clustering.py`\n- `src/postprocess.py`")
    with col_stat4:
        st.info("**Module 4: Benchmark & Y bạ**\n- `src/benchmark.py`\n- `src/pipeline.py`")

if input_image is None:
    st.info("👋 Vui lòng chọn hoặc tải ảnh ở thanh công cụ bên trái để bắt đầu phân tích.")
    st.stop()

# -----------------------------------------------------------------------------
# TAB NỘI DUNG CHÍNH
# -----------------------------------------------------------------------------
tab_pipeline, tab_benchmark, tab_matrix = st.tabs(
    ["🚀 Quy trình Phân đoạn & Chẩn đoán", "📊 Đối sánh định lượng (Benchmark)", "🔢 Dữ liệu Ma trận điểm ảnh"]
)

with tab_pipeline:
    st.subheader("1. Ảnh đầu vào & Tiền xử lý (Preprocessing)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(input_image, caption=f"Ảnh gốc (Shape: {input_image.shape})", use_container_width=True)

    with col2:
        st.markdown("**Ảnh mức xám (Grayscale Scratch):**")
        try:
            from src.preprocess import rgb_to_gray_scratch

            gray_img = rgb_to_gray_scratch(input_image)
            st.image(gray_img, caption="Grayscale (0.299R + 0.587G + 0.114B)", use_container_width=True)
        except NotImplementedError as e:
            st.markdown(
                '<div class="skeleton-notice">⚠️ <b>src/preprocess.py:</b> <code>rgb_to_gray_scratch</code> '
                'chưa được triển khai trên nhánh này. (Chờ merge PR nhánh <code>feat/preprocess</code>)</div>',
                unsafe_allow_html=True,
            )
            # Giả lập tạm để hiển thị giao diện không bị gián đoạn
            gray_img = (
                0.299 * input_image[:, :, 0] + 0.587 * input_image[:, :, 1] + 0.114 * input_image[:, :, 2]
            ).astype(np.uint8)

    with col3:
        st.markdown("**Ảnh sau làm mịn (Gaussian Filter Scratch):**")
        try:
            from src.preprocess import gaussian_filter_scratch

            blurred_img = gaussian_filter_scratch(gray_img, kernel_size=blur_kernel, sigma=blur_sigma)
            st.image(blurred_img, caption=f"Gauss Blur (Kernel {blur_kernel}x{blur_kernel})", use_container_width=True)
        except NotImplementedError as e:
            st.markdown(
                '<div class="skeleton-notice">⚠️ <b>src/preprocess.py:</b> <code>gaussian_filter_scratch</code> '
                'chưa được triển khai.</div>',
                unsafe_allow_html=True,
            )
            blurred_img = gray_img

    st.markdown("---")
    st.subheader("2. Phân tầng 1: Tách nền bằng Otsu Thresholding (Scratch)")
    col_otsu1, col_otsu2 = st.columns(2)
    leaf_mask = None
    try:
        from src.otsu import otsu_threshold_scratch

        otsu_res = otsu_threshold_scratch(blurred_img)
        leaf_mask = otsu_res["binary_mask"]
        with col_otsu1:
            st.image(leaf_mask, caption=f"Mặt nạ lá (Otsu T* = {otsu_res['threshold']})", use_container_width=True)
        with col_otsu2:
            st.line_chart(otsu_res["variance_curve"], height=220)
            st.caption("Đường cong phương sai giữa các lớp sigma_B^2(t)")
    except NotImplementedError:
        with col_otsu1:
            st.markdown(
                '<div class="skeleton-notice">⚠️ <b>src/otsu.py:</b> <code>otsu_threshold_scratch</code> '
                'chưa được triển khai trên nhánh này.</div>',
                unsafe_allow_html=True,
            )
            # Tạo mask giả lập để các bước sau vẫn minh họa được
            leaf_mask = np.where(gray_img < 230, 255, 0).astype(np.uint8)
            st.image(leaf_mask, caption="Mặt nạ phiến lá (Mock ROI)", use_container_width=True)

    st.markdown("---")
    st.subheader("3. Phân tầng 2: Phân cụm màu K-Means & Tách vết bệnh")
    col_km1, col_km2 = st.columns(2)
    disease_mask = None
    try:
        from src.kmeans import kmeans_scratch
        from src.color_clustering import extract_disease_and_healthy_masks

        kmeans_res = kmeans_scratch(input_image, k=k_clusters, roi_mask=leaf_mask)
        color_res = extract_disease_and_healthy_masks(
            input_image, kmeans_res["labels_2d"], kmeans_res["centroids"], leaf_mask=leaf_mask
        )
        disease_mask = color_res["disease_mask"]

        with col_km1:
            st.image(kmeans_res["segmented_img"], caption=f"K-Means (k={k_clusters})", use_container_width=True)
        with col_km2:
            st.image(disease_mask, caption="Mặt nạ vết bệnh (Disease Mask)", use_container_width=True)
    except NotImplementedError:
        st.markdown(
            '<div class="skeleton-notice">⚠️ <b>src/kmeans.py</b> hoặc <b>src/color_clustering.py:</b> '
            'Chức năng phân cụm K-means hoặc tách màu đang chờ hoàn thiện trên nhánh tính năng.</div>',
            unsafe_allow_html=True,
        )
        # Giả lập vết bệnh
        disease_mask = np.zeros_like(leaf_mask)
        yy, xx = np.ogrid[: leaf_mask.shape[0], : leaf_mask.shape[1]]
        disease_mask[((xx - 110) ** 2 + (yy - 100) ** 2) <= 16 ** 2] = 255
        disease_mask[((xx - 150) ** 2 + (yy - 160) ** 2) <= 12 ** 2] = 255

    st.markdown("---")
    st.subheader("4. Hậu xử lý Hình thái học & Phát hiện ổ bệnh (BFS)")
    col_post1, col_post2 = st.columns(2)
    cleaned_mask = disease_mask
    bboxes = []
    try:
        from src.postprocess import closing_scratch, connected_components_bfs_scratch, opening_scratch

        morphed = opening_scratch(disease_mask, kernel_size=morph_kernel)
        morphed = closing_scratch(morphed, kernel_size=morph_kernel)
        bfs_res = connected_components_bfs_scratch(morphed, min_area=min_lesion_area)
        cleaned_mask = bfs_res["cleaned_mask"]
        bboxes = bfs_res["bboxes"]

        with col_post1:
            st.image(cleaned_mask, caption=f"Mặt nạ khử nhiễu (Đếm được {bfs_res['num_lesions']} ổ bệnh)", use_container_width=True)
        with col_post2:
            annotated_img = draw_bounding_boxes(input_image, bboxes)
            st.image(annotated_img, caption="Đánh dấu hộp bao Bounding Box ổ bệnh", use_container_width=True)
    except NotImplementedError:
        st.markdown(
            '<div class="skeleton-notice">⚠️ <b>src/postprocess.py:</b> Các hàm hình thái học hoặc BFS '
            'chưa được triển khai trên nhánh này.</div>',
            unsafe_allow_html=True,
        )
        bboxes = [(94, 84, 32, 32, 800), (138, 148, 24, 24, 450)]
        annotated_mock = draw_bounding_boxes(input_image, bboxes)
        with col_post1:
            st.image(disease_mask, caption="Mặt nạ bệnh (Chưa qua lọc morphology)", use_container_width=True)
        with col_post2:
            st.image(annotated_mock, caption="Minh họa BBox hộp bao ổ bệnh (Mock)", use_container_width=True)

    st.markdown("---")
    st.subheader("5. Kết quả Chẩn đoán Y bạ Cây trồng (Health Diagnosis Report)")
    try:
        from src.benchmark import diagnose_leaf_severity

        diag = diagnose_leaf_severity(leaf_mask, cleaned_mask, len(bboxes))
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Cấp độ bệnh", f"Cấp {diag['severity_level']}")
        col_m2.metric("Tỷ lệ tổn thương", f"{diag['disease_ratio_percent']:.2f}%")
        col_m3.metric("Số ổ bệnh phát hiện", f"{diag['num_lesions']}")
        col_m4.metric("Diện tích phiến lá", f"{diag['leaf_area_px']:,} px")

        st.success(f"**Chẩn đoán:** {diag['severity_label']}")
        st.info(f"💡 **Khuyến nghị điều trị:** {diag['recommendation']}")
    except NotImplementedError:
        st.markdown(
            '<div class="skeleton-notice">⚠️ <b>src/benchmark.py:</b> <code>diagnose_leaf_severity</code> '
            'chưa được triển khai.</div>',
            unsafe_allow_html=True,
        )
        st.warning("Đang hiển thị mẫu chẩn đoán: Cấp 2 (Nhiễm bệnh trung bình) - Tỷ lệ ~7.5%.")

with tab_benchmark:
    st.subheader("📊 Bảng đối sánh định lượng: Thuật toán Scratch vs OpenCV")
    st.markdown(
        "Bảng theo dõi và so sánh thời gian thực thi (Runtime) cũng như độ tương đồng (IoU, Dice) "
        "giữa thuật toán tự phát triển từ đầu và thư viện C++ OpenCV tiêu chuẩn."
    )
    benchmark_data = [
        {"Tác vụ": "Tách nền Otsu", "Module Scratch (ms)": "Đang đo...", "OpenCV Baseline (ms)": "Đang đo...", "IoU Khớp": "--", "Dice Score": "--"},
        {"Tác vụ": "Phân cụm K-Means (k=3)", "Module Scratch (ms)": "Đang đo...", "OpenCV Baseline (ms)": "Đang đo...", "IoU Khớp": "--", "Dice Score": "--"},
    ]
    st.table(pd.DataFrame(benchmark_data))
    st.caption("Ghi chú: Khi các nhánh tính năng được merge vào main, bảng này sẽ tự động cập nhật số liệu thời gian thực.")

with tab_matrix:
    st.subheader("🔢 Khảo sát ma trận điểm ảnh số (Pixel Data Matrix)")
    st.markdown("Xem giá trị số học chi tiết của ma trận mức xám và xuất dữ liệu sang định dạng CSV:")
    sample_sub_matrix = gray_img[:16, :16]
    st.dataframe(pd.DataFrame(sample_sub_matrix))

    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, gray_img, fmt="%d", delimiter=",")
    st.download_button(
        label="📥 Tải xuống toàn bộ ma trận ảnh (.csv)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="leaf_gray_matrix.csv",
        mime="text/csv",
    )
