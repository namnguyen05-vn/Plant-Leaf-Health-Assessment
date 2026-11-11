"""LeafGuard AI — Plant Leaf Health Assessment Dashboard.

Giao diện SaaS / AgriTech hiện đại chuẩn Figma kết nối với `src/pipeline.py`,
`src/postprocess.py`, `src/benchmark.py` và các module xử lý ảnh `src/`.
"""

import io
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.data_io import matrix_to_csv_bytes
from src.postprocess import (
    closing_scratch,
    draw_lesion_bboxes_scratch,
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
# CẤU HÌNH TRANG & FIGMA DESIGN SYSTEM CSS
# =============================================================================
st.set_page_config(
    page_title="LeafGuard AI — Plant Health Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

_ST_MAJOR, _ST_MINOR = (int(x) for x in st.__version__.split(".")[:2])
_STRETCH_KW: Dict[str, Any] = (
    {"width": "stretch"} if (_ST_MAJOR, _ST_MINOR) >= (1, 50) else {"use_container_width": True}
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root, html, body, .stApp {
        color-scheme: light !important;
        --text-color: #0F172A;
        --background-color: #F8FAFC;
        --secondary-background-color: #FFFFFF;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }

    /* Nền chính Slate 50 */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Sidebar trắng + viền phải Slate 200 */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
    }

    /* Giảm khoảng trống trên cùng & giữ icon thanh Header luôn rõ */
    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1380px !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.88) !important;
        backdrop-filter: blur(8px);
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {
        color: #0F172A !important;
        fill: #0F172A !important;
    }

    /* Ép tương phản rõ cho mọi văn bản, nhãn widget và chú thích ảnh */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label *,
    [data-testid="stText"] {
        color: #0F172A !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stImageCaption"],
    [data-testid="stImageCaption"] * {
        color: #475569 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }

    /* Selectbox & Popover Dropdowns */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="select"] input,
    [data-baseweb="select"] svg {
        color: #0F172A !important;
        fill: #0F172A !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10) !important;
    }
    ul[role="listbox"] li,
    ul[role="listbox"] li * {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li:hover *,
    ul[role="listbox"] li[aria-selected="true"],
    ul[role="listbox"] li[aria-selected="true"] * {
        background-color: #ECFDF5 !important;
        color: #047857 !important;
    }

    /* Expander, Sliders & File Uploader */
    [data-testid="stExpander"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] svg {
        color: #0F172A !important;
        fill: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] span {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 1.5px dashed #94A3B8 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"] small {
        color: #334155 !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
        font-weight: 600 !important;
    }

    /* Figma Card Surface cho st.container(border=True) */
    [class*="st-key-card_"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
        padding: 1rem 1.15rem !important;
        color: #0F172A !important;
    }

    /* Pill / Segmented Control Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #E2E8F0 !important;
        padding: 4px !important;
        border-radius: 12px !important;
        gap: 4px !important;
        display: inline-flex !important;
        margin-bottom: 0.75rem !important;
    }
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] * {
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #475569 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }
    button[data-baseweb="tab"] {
        padding: 8px 18px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #047857 !important;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Nút bấm & Nút tải xuống (đặt sau rule p để bảo vệ chữ trắng) */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button,
    div.stButton > button * {
        color: #0F172A !important;
    }
    div.stButton > button:hover {
        border-color: #059669 !important;
        background-color: #ECFDF5 !important;
    }
    div.stButton > button:hover,
    div.stButton > button:hover * {
        color: #047857 !important;
    }
    div.stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        background-color: #059669 !important;
        border: 1px solid #059669 !important;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.18) !important;
        transition: all 0.15s ease !important;
    }
    div.stDownloadButton > button,
    div.stDownloadButton > button * {
        color: #FFFFFF !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #047857 !important;
        border-color: #047857 !important;
    }
    div.stDownloadButton > button:hover,
    div.stDownloadButton > button:hover * {
        color: #FFFFFF !important;
    }

    /* Sidebar Brand Logo */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 6px 2px 14px 2px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 14px;
    }
    .brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #FFFFFF !important;
        flex-shrink: 0;
        box-shadow: 0 4px 10px rgba(5, 150, 105, 0.22);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .brand-sub {
        font-size: 0.74rem;
        font-weight: 500;
        color: #475569 !important;
        margin-top: 2px;
    }

    /* Sidebar section label */
    .sb-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569 !important;
        margin: 12px 0 6px 0;
    }

    /* Hero Diagnostic Banner */
    .hero-banner {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .hero-meta-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .hero-meta-chips {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 0.82rem;
        color: #334155 !important;
        font-weight: 500;
    }
    .meta-chip {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
    }
    .meta-chip b { color: #0F172A !important; }
    .hero-right {
        min-width: 280px;
        flex: 0 0 auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .status-pill {
        padding: 7px 15px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    /* Bảng màu 5 cấp độ chuẩn Figma */
    .sev-0 { background-color: #DCFCE7 !important; color: #166534 !important; border: 1px solid #86EFAC !important; }
    .sev-1 { background-color: #FEF9C3 !important; color: #854D0E !important; border: 1px solid #FDE047 !important; }
    .sev-2 { background-color: #FFEDD5 !important; color: #9A3412 !important; border: 1px solid #FDBA74 !important; }
    .sev-3 { background-color: #FEE2E2 !important; color: #991B1B !important; border: 1px solid #FCA5A5 !important; }
    .sev-4 { background-color: #F3E8FF !important; color: #6B21A8 !important; border: 1px solid #D8B4FE !important; }

    .prog-wrap {
        width: 100%;
        min-width: 260px;
    }
    .prog-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155 !important;
        margin-bottom: 4px;
    }
    .prog-track {
        width: 100%;
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
    }
    .prog-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.3s ease;
    }

    /* 4-Column KPI Metric Card */
    .kpi-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569 !important;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.78rem;
        font-weight: 600;
        color: #047857 !important;
        margin-top: 6px;
    }

    /* Card Title & Clinical Recommendation */
    .card-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .clinical-box {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 5px solid #059669 !important;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin: 8px 0 14px 0;
    }
    .clinical-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-bottom: 4px;
    }
    .clinical-text {
        font-size: 0.88rem;
        color: #1E293B !important;
        line-height: 1.5;
    }

    /* Module Status Footer */
    .status-footer {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        padding: 10px 12px;
        margin-top: 12px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        padding: 3px 0;
        color: #1E293B !important;
    }
    .status-row code {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        padding: 2px 6px !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
        font-size: 0.73rem !important;
    }
    .dot-ok { color: #047857 !important; font-weight: 700 !important; }
    .dot-fb { color: #B45309 !important; font-weight: 700 !important; }

    /* Đảm bảo st.info / st.warning luôn có chữ tối dễ đọc */
    [data-testid="stAlert"] {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] * {
        color: #0F172A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE & GIÁ TRỊ MẶC ĐỊNH
# =============================================================================
_DEFAULT_PARAMS: Dict[str, Any] = {
    "blur_ksize": 5,
    "blur_sigma": 1.0,
    "k_clusters": 4,
    "halo_sensitivity": 0.65,
    "morph_ksize": 5,
    "min_area": 20,
}

for _k, _v in _DEFAULT_PARAMS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_default_params() -> None:
    """Khôi phục các siêu tham số thuật toán về mặc định."""
    for _k, _v in _DEFAULT_PARAMS.items():
        st.session_state[_k] = _v


# =============================================================================
# HÀM TIỆN ÍCH XỬ LÝ & TRỰC QUAN HÓA
# =============================================================================

def create_sample_leaf_image(preset: str = "halo") -> np.ndarray:
    """Tạo nhanh ảnh mẫu lá cây thực địa giả lập ngay trên bộ nhớ."""
    H, W = 320, 320
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    if preset == "field":
        img[:] = [230, 232, 228]
        img[(xx + 0.75 * yy) < 135] = [142, 102, 65]
        img[(xx < 95) & (yy > 205)] = [160, 122, 78]

        leaf_mask = ((xx - 175) / 92.0) ** 2 + ((yy - 160) / 122.0) ** 2 <= 1.0
        img[leaf_mask] = [38, 138, 68]

        glare = (((xx - 165) / 36.0) ** 2 + ((yy - 108) / 26.0) ** 2 <= 1.0) & leaf_mask
        img[glare] = [205, 228, 198]

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

    img[:] = [238, 240, 238]
    leaf_mask = ((xx - 160) / 98.0) ** 2 + ((yy - 160) / 128.0) ** 2 <= 1.0
    img[leaf_mask] = [42, 142, 74]

    main_vein = (np.abs(xx - 160) <= 2) & leaf_mask
    img[main_vein] = [88, 185, 118]
    for offset_y in (-55, -15, 25, 65):
        side_vein = (np.abs((yy - (160 + offset_y)) - 0.35 * np.abs(xx - 160)) <= 1.5) & leaf_mask
        img[side_vein] = [78, 175, 108]

    halo_lesions = [
        (126, 115, 24, [(-6, -5, 4), (5, -2, 4), (-2, 6, 5), (7, 5, 3)]),
        (196, 178, 20, [(-5, -3, 4), (4, 3, 4), (0, -6, 3)]),
        (134, 214, 16, [(-3, -2, 4), (4, 2, 3)]),
    ]
    for cx, cy, hr, dots in halo_lesions:
        halo_outer = (((xx - cx) ** 2 + (yy - cy) ** 2) <= hr ** 2) & leaf_mask
        img[halo_outer] = [196, 198, 52]
        for dx, dy, dr in dots:
            dot = (((xx - (cx + dx)) ** 2 + (yy - (cy + dy)) ** 2) <= dr ** 2) & leaf_mask
            img[dot] = [102, 54, 22]

    img[75:77, 155:157] = [145, 75, 25]
    return img


def render_annotated_overlay(
    rgb_img: np.ndarray,
    disease_mask: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, int]],
) -> np.ndarray:
    """Tô lớp phủ bán trong suốt lên vùng bệnh và vẽ Bounding Box đánh số ổ bệnh bằng NumPy."""
    overlay = rgb_img.astype(np.float32).copy()
    mask_bool = disease_mask > 0
    tint = np.array([239.0, 68.0, 68.0], dtype=np.float32)
    overlay[mask_bool] = 0.62 * overlay[mask_bool] + 0.38 * tint
    blended = np.clip(np.round(overlay), 0, 255).astype(np.uint8)
    return draw_lesion_bboxes_scratch(
        blended, bboxes, box_color=(220, 38, 38), thickness=2, draw_labels=True
    )


def build_binary_difference_map(mask_scratch: np.ndarray, mask_cv: np.ndarray) -> np.ndarray:
    """Tạo bản đồ sai khác trực quan: Khớp tiền cảnh (Xanh), Chỉ Scratch (Đỏ), Chỉ OpenCV (Lam)."""
    H, W = mask_scratch.shape[:2]
    s = mask_scratch > 0
    c = mask_cv > 0
    heatmap = np.full((H, W, 3), [241, 245, 249], dtype=np.uint8)
    heatmap[s & c] = [16, 185, 129]     # Trùng khớp (Emerald)
    heatmap[s & (~c)] = [239, 68, 68]   # Chỉ Scratch (Red)
    heatmap[(~s) & c] = [59, 130, 246]  # Chỉ OpenCV (Blue)
    return heatmap


def build_rgb_difference_heatmap(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Tạo bản đồ nhiệt sai khác giữa hai ảnh phân cụm K-means."""
    diff = np.mean(np.abs(img_a.astype(np.float32) - img_b.astype(np.float32)), axis=-1)
    norm = np.clip(diff / max(float(diff.max()), 1.0), 0.0, 1.0)
    cmap = plt.get_cmap("magma")
    rgba = cmap(norm)
    return np.clip(np.round(rgba[:, :, :3] * 255.0), 0, 255).astype(np.uint8)


def image_to_png_bytes(img_array: np.ndarray) -> bytes:
    """Chuyển đổi mảng NumPy sang bytes PNG."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def safe_matrix_to_csv_bytes(matrix_2d: np.ndarray) -> bytes:
    """Xuất ma trận 2D sang CSV bytes (dùng `data_io` hoặc fallback nội bộ)."""
    try:
        return matrix_to_csv_bytes(matrix_2d)
    except NotImplementedError:
        buf = io.StringIO()
        np.savetxt(buf, matrix_2d, fmt="%d", delimiter=",")
        return buf.getvalue().encode("utf-8")


def plot_figma_otsu_chart(
    histogram: np.ndarray,
    variance_curve: np.ndarray,
    threshold_scratch: int,
    threshold_opencv: int,
) -> plt.Figure:
    """Vẽ biểu đồ Histogram 256 mức xám và đường cong sigma_B^2(t) đồng bộ tông màu sáng Figma."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    x = np.arange(256)

    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(colors="#475569", labelsize=8.5)
        ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.8)

    ax1.bar(x, histogram, color="#059669", alpha=0.80, width=1.0, label="Histogram (256 mức)")
    ax1.axvline(threshold_scratch, color="#DC2626", linestyle="--", linewidth=1.8, label=f"T* Scratch = {threshold_scratch}")
    ax1.axvline(threshold_opencv, color="#2563EB", linestyle=":", linewidth=1.8, label=f"T* OpenCV = {threshold_opencv}")
    ax1.set_title("Phân bố Histogram & Ngưỡng cắt Otsu", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)
    ax1.set_xlabel("Cường độ (0..255)", fontsize=9, color="#475569")
    ax1.set_ylabel("Số lượng pixel", fontsize=9, color="#475569")
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8)

    ax2.plot(x, variance_curve, color="#D97706", linewidth=2.0, label=r"$\sigma_B^2(t)$")
    ax2.axvline(threshold_scratch, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Đỉnh cực đại T* = {threshold_scratch}")
    ax2.set_title(r"Đường cong Phương sai Giữa các lớp $\sigma_B^2(t)$", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)
    ax2.set_xlabel("Ngưỡng thử t (0..255)", fontsize=9, color="#475569")
    ax2.set_ylabel(r"$\sigma_B^2(t)$", fontsize=9, color="#475569")
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8)

    plt.tight_layout()
    return fig


def render_kpi_card(label: str, value: str, sub: str) -> None:
    """Render 1 thẻ KPI Figma đồng nhất."""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_hero_banner(
    file_name: str,
    shape: Tuple[int, ...],
    exec_ms: float,
    severity_level: int,
    severity_label: str,
    ratio_pct: float,
) -> None:
    """Render Top Hero Header & Diagnostic Banner."""
    h, w = shape[0], shape[1]
    bar_colors = {0: "#10B981", 1: "#EAB308", 2: "#F97316", 3: "#EF4444", 4: "#9333EA"}
    fill_color = bar_colors.get(severity_level, "#059669")
    fill_width = 0.0 if ratio_pct <= 0.0 else min(100.0, max(2.0, ratio_pct))

    st.markdown(
        f'<div class="hero-banner">'
        f'  <div>'
        f'    <div class="hero-meta-title">🌿 Báo Cáo Phân Tích Sức Khỏe Lá Cây</div>'
        f'    <div class="hero-meta-chips">'
        f'      <span class="meta-chip">📄 <b>{file_name}</b></span>'
        f'      <span class="meta-chip">📐 {w} × {h} px</span>'
        f'      <span class="meta-chip">⚡ {exec_ms:.1f} ms</span>'
        f'    </div>'
        f'  </div>'
        f'  <div class="hero-right">'
        f'    <div class="status-pill sev-{severity_level}">● {severity_label}</div>'
        f'    <div class="prog-wrap">'
        f'      <div class="prog-header"><span>Tỷ lệ diện tích lá nhiễm bệnh</span><span>{ratio_pct:.2f}%</span></div>'
        f'      <div class="prog-track"><div class="prog-fill" style="width:{fill_width:.1f}%; background-color:{fill_color};"></div></div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_metrics_summary_df(
    metrics: Dict[str, float],
    time_scratch_ms: float,
    time_cv_ms: float,
) -> pd.DataFrame:
    """Tạo bảng tổng hợp các chỉ số định lượng Scratch vs OpenCV."""
    return pd.DataFrame(
        [
            {"Chỉ số định lượng": "IoU (Jaccard Index)", "Giá trị đo": f"{metrics['iou_pct']:.2f}%"},
            {"Chỉ số định lượng": "Dice Score (F1)", "Giá trị đo": f"{metrics['dice_pct']:.2f}%"},
            {"Chỉ số định lượng": "Pixel Accuracy", "Giá trị đo": f"{metrics['pixel_accuracy_pct']:.2f}%"},
            {"Chỉ số định lượng": "MSE (Mean Squared Error)", "Giá trị đo": f"{metrics['mse']:.4f}"},
            {"Chỉ số định lượng": "PSNR (dB)", "Giá trị đo": f"{metrics['psnr']:.2f} dB"},
            {"Chỉ số định lượng": "Execution Time (Scratch / OpenCV)", "Giá trị đo": f"{time_scratch_ms:.2f} ms / {time_cv_ms:.2f} ms"},
        ]
    )


# =============================================================================
# THANH SIDEBAR (LEFT CONTROL PANEL)
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="brand-box">'
        '  <div class="brand-icon">🌿</div>'
        '  <div>'
        '    <div class="brand-title">LeafGuard AI</div>'
        '    <div class="brand-sub">Multi-Stage Segmentation &amp; Diagnosis</div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Khu vực 1 — Nguồn dữ liệu đầu vào
    st.markdown('<div class="sb-label">1. Nguồn dữ liệu đầu vào</div>', unsafe_allow_html=True)
    input_choice = st.radio(
        "Nguồn dữ liệu:",
        ["Dùng ảnh mẫu thử nghiệm", "Tải file từ máy (.JPG/.PNG/.CSV)"],
        label_visibility="collapsed",
    )

    raw_input_source: Any = None
    source_name = "sample_leaf.png"

    if input_choice == "Tải file từ máy (.JPG/.PNG/.CSV)":
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc chọn ảnh / ma trận CSV",
            type=["png", "jpg", "jpeg", "csv"],
        )
        if uploaded_file is not None:
            raw_input_source = uploaded_file
            source_name = uploaded_file.name
    else:
        sample_preset = st.selectbox(
            "Chọn ảnh mẫu thực địa:",
            [
                "Mẫu 1: Đốm tâm nâu & quầng vàng lớn",
                "Mẫu 2: Lá bóng nắng, đốm nhỏ (<3%) & nền đất nâu",
            ],
        )
        preset_key = "halo" if sample_preset.startswith("Mẫu 1") else "field"
        raw_input_source = create_sample_leaf_image(preset=preset_key)
        source_name = f"sample_{preset_key}.png"

    # Khu vực 2 — Chế độ phân tích
    st.markdown('<div class="sb-label">2. Chế độ phân tích</div>', unsafe_allow_html=True)
    analysis_mode = st.radio(
        "Chế độ:",
        [
            "Chẩn đoán Đa tầng (Full Pipeline)",
            "Chuyên sâu Otsu (Scratch vs OpenCV)",
            "Chuyên sâu K-means (Scratch vs OpenCV)",
        ],
        label_visibility="collapsed",
    )

    # Khu vực 3 — Tham số thuật toán (Gom gọn vào Expander)
    st.markdown('<div class="sb-label">3. Cấu hình thuật toán</div>', unsafe_allow_html=True)
    with st.expander("⚙️ Tinh chỉnh tham số thuật toán", expanded=False):
        blur_ksize = st.select_slider("Kích thước lọc Gaussian:", options=[3, 5, 7, 9, 11], key="blur_ksize")
        blur_sigma = st.slider("Gaussian Sigma:", min_value=0.5, max_value=3.0, step=0.1, key="blur_sigma")
        k_clusters = st.slider("Số cụm K-means (K):", min_value=2, max_value=6, step=1, key="k_clusters")
        halo_sensitivity = st.slider(
            "Độ nhạy quầng vàng (Halo):",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="halo_sensitivity",
        )
        morph_ksize = st.select_slider("Kernel Hình thái học (Closing):", options=[3, 5, 7, 9], key="morph_ksize")
        min_area = st.slider("Diện tích ổ bệnh tối thiểu (px):", min_value=5, max_value=200, step=5, key="min_area")
        st.button("↺ Khôi phục mặc định", on_click=_reset_default_params, **_STRETCH_KW)

    status_placeholder = st.empty()


# =============================================================================
# KIỂM TRA ĐẦU VÀO TRƯỚC KHI CHẠY
# =============================================================================
if raw_input_source is None:
    st.container(border=True).info(
        "👋 **Chưa có dữ liệu đầu vào.** Vui lòng tải file ảnh (`.jpg`, `.png`) hoặc ma trận (`.csv`) ở bảng điều khiển bên trái."
    )
    st.stop()


# =============================================================================
# THỰC THI PIPELINE TOÀN TRÌNH (CHUNG CHO CÁC CHỈ SỐ BANNER & KPI)
# =============================================================================
t_pipe_start = time.perf_counter()
result = run_full_diagnosis_pipeline(
    file_source=raw_input_source,
    k_clusters=k_clusters,
    blur_kernel=blur_ksize,
    blur_sigma=blur_sigma,
    min_lesion_area=min_area,
    morph_kernel=morph_ksize,
    yellow_halo_sensitivity=halo_sensitivity,
    run_benchmark_comparison=True,
)
total_exec_ms = (time.perf_counter() - t_pipe_start) * 1000.0

fallback_set = set(result.get("fallback_modules", []))

# Cập nhật Khu vực 4 Sidebar: System Status Footer
with status_placeholder.container():
    st.markdown('<div class="sb-label">4. Trạng thái Module</div>', unsafe_allow_html=True)
    modules_status = [
        ("postprocess / benchmark", False),
        ("pipeline orchestrator", False),
        ("data_io", "data_io" in fallback_set),
        ("preprocess", "preprocess" in fallback_set),
        ("otsu", "otsu" in fallback_set),
        ("kmeans", "kmeans" in fallback_set),
        ("color_clustering", "color_clustering" in fallback_set),
    ]
    rows_html = "".join(
        f'<div class="status-row"><span><code>{name}</code></span>'
        f'<span class="{"dot-fb" if is_fb else "dot-ok"}">{"● Fallback" if is_fb else "● From Scratch"}</span></div>'
        for name, is_fb in modules_status
    )
    st.markdown(f'<div class="status-footer">{rows_html}</div>', unsafe_allow_html=True)


# Chuẩn bị các biến dùng chung cho Main Workspace
diag = result["diagnosis"]
bm = result["benchmark"]
annotated_overlay = render_annotated_overlay(
    result["original_rgb"],
    result["postprocess_result"]["cleaned_mask"],
    result["postprocess_result"]["bboxes"],
)


# =============================================================================
# MAIN WORKSPACE — PHẦN A: TOP HERO HEADER & DIAGNOSTIC BANNER
# =============================================================================
render_hero_banner(
    file_name=source_name,
    shape=result["original_rgb"].shape,
    exec_ms=total_exec_ms,
    severity_level=diag["severity_level"],
    severity_label=diag["severity_label"],
    ratio_pct=diag["disease_ratio_percent"],
)


# =============================================================================
# CHẾ ĐỘ 1: CHẨN ĐOÁN ĐA TẦNG (FULL PIPELINE)
# =============================================================================
if analysis_mode == "Chẩn đoán Đa tầng (Full Pipeline)":
    km_bm = bm["kmeans_disease_metrics"]
    otsu_bm = bm["otsu_metrics"]

    # PHẦN B: Hàng 4 Thẻ Chỉ số KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Tỷ lệ nhiễm bệnh (%)",
            f"{diag['disease_ratio_percent']:.2f}%",
            f"Bệnh: {diag['disease_area_px']:,} / Lá: {diag['leaf_area_px']:,} px",
        )
    with k2:
        render_kpi_card(
            "Số ổ bệnh phát hiện",
            f"{diag['num_lesions']} ổ",
            f"Đã lọc nhiễu < {min_area} px",
        )
    with k3:
        render_kpi_card(
            "Độ tương đồng (IoU Score)",
            f"{km_bm['iou_pct']:.1f}%",
            f"Dice: {km_bm['dice_pct']:.1f}% (vs OpenCV)",
        )
    with k4:
        render_kpi_card(
            "Sai số & Tốc độ (PSNR / Time)",
            f"{km_bm['psnr']:.1f} dB",
            f"MSE: {km_bm['mse']:.2f} | {total_exec_ms:.1f} ms",
        )

    # PHẦN C: Không gian làm việc chia 3 Tab
    tab_diag, tab_stages, tab_bench = st.tabs(
        [
            "🔍 Kết quả Chẩn đoán & Khoanh vùng",
            "🧩 Chi tiết Pipeline 4 Giai đoạn",
            "📈 Đối sánh Định lượng (From Scratch vs OpenCV)",
        ]
    )

    # --- TAB 1: KẾT QUẢ CHẨN ĐOÁN & KHOANH VÙNG ---
    with tab_diag:
        col_left, col_right = st.columns(2)
        with col_left:
            with st.container(border=True, key="card_t1_orig"):
                st.markdown('<div class="card-title">🖼️ Ảnh gốc ban đầu (Original Image)</div>', unsafe_allow_html=True)
                st.image(result["original_rgb"], **_STRETCH_KW)
        with col_right:
            with st.container(border=True, key="card_t1_annot"):
                st.markdown(
                    '<div class="card-title">🎯 Kết quả Chẩn đoán &amp; Khoanh vùng (Annotated Diagnosis Output)</div>',
                    unsafe_allow_html=True,
                )
                st.image(annotated_overlay, **_STRETCH_KW)

        st.markdown(
            f'<div class="clinical-box">'
            f'  <div class="clinical-title">🩺 Nhận xét lâm sàng &amp; Khuyến nghị — {diag["severity_label"]}</div>'
            f'  <div class="clinical-text">{diag["recommendation"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Tải ảnh kết quả (.PNG)",
                data=image_to_png_bytes(annotated_overlay),
                file_name="leafguard_diagnosis.png",
                mime="image/png",
                **_STRETCH_KW,
            )
        with dl_col2:
            st.download_button(
                label="📊 Xuất ma trận phân đoạn (.CSV)",
                data=safe_matrix_to_csv_bytes(result["postprocess_result"]["cleaned_mask"]),
                file_name="leafguard_disease_mask.csv",
                mime="text/csv",
                **_STRETCH_KW,
            )

    # --- TAB 2: CHI TIẾT PIPELINE 4 GIAI ĐOẠN ---
    with tab_stages:
        st_c1, st_c2, st_c3, st_c4 = st.columns(4)
        with st_c1:
            with st.container(border=True, key="card_t2_s1"):
                st.markdown('<div class="card-title">1️⃣ Otsu ROI (Tách lá)</div>', unsafe_allow_html=True)
                st.image(result["leaf_mask"], caption=f"T* = {result['otsu_result']['threshold']} + Lấp lỗ BFS", **_STRETCH_KW)
        with st_c2:
            with st.container(border=True, key="card_t2_s2"):
                st.markdown('<div class="card-title">2️⃣ Phân cụm K-means</div>', unsafe_allow_html=True)
                st.image(
                    result["kmeans_result"]["segmented_img"],
                    caption=f"K={k_clusters}, {result['kmeans_result']['n_iter']} iter",
                    **_STRETCH_KW,
                )
        with st_c3:
            with st.container(border=True, key="card_t2_s3"):
                st.markdown('<div class="card-title">3️⃣ Hình thái học</div>', unsafe_allow_html=True)
                st.image(
                    result["morphed_disease_mask"],
                    caption=f"Closing(k={morph_ksize}) -> Opening",
                    **_STRETCH_KW,
                )
        with st_c4:
            with st.container(border=True, key="card_t2_s4"):
                st.markdown('<div class="card-title">4️⃣ Liên thông BFS</div>', unsafe_allow_html=True)
                st.image(
                    annotated_overlay,
                    caption=f"Phát hiện {diag['num_lesions']} ổ bệnh",
                    **_STRETCH_KW,
                )

        with st.container(border=True, key="card_t2_table"):
            st.markdown('<div class="card-title">📋 Danh sách tọa độ &amp; Diện tích từng ổ bệnh</div>', unsafe_allow_html=True)
            bboxes = result["postprocess_result"]["bboxes"]
            if bboxes:
                leaf_area = max(diag["leaf_area_px"], 1)
                bbox_df = pd.DataFrame(
                    [
                        {
                            "ID": f"#{idx}",
                            "x": b[0],
                            "y": b[1],
                            "w": b[2],
                            "h": b[3],
                            "Diện tích (px)": b[4],
                            "Tỷ lệ trên lá (%)": round((b[4] / leaf_area) * 100.0, 3),
                        }
                        for idx, b in enumerate(bboxes, start=1)
                    ]
                )
                st.dataframe(bbox_df, **_STRETCH_KW, height=220, hide_index=True)
            else:
                st.info("Không có ổ bệnh nào đạt ngưỡng diện tích tối thiểu.")

    # --- TAB 3: ĐỐI SÁNH ĐỊNH LƯỢNG (FROM SCRATCH VS OPENCV) ---
    with tab_bench:
        cmp_target = st.radio(
            "Chọn tầng thuật toán đối sánh:",
            ["Tầng 1: Tách nền Otsu (Scratch vs cv2.threshold)", "Tầng 2: Phân đoạn vùng bệnh K-means (Scratch vs cv2.kmeans)"],
            horizontal=True,
        )

        if cmp_target.startswith("Tầng 1"):
            mask_s = result["otsu_result"]["binary_mask"]
            mask_c = bm["otsu_cv"]["binary_mask"]
            diff_vis = build_binary_difference_map(mask_s, mask_c)
            m_data = otsu_bm
            t_s = result["otsu_result"]["exec_time_ms"]
            t_c = bm["otsu_cv"]["exec_time_ms"]
            t_lbl_s = f"Otsu Scratch (T* = {result['otsu_result']['threshold']})"
            t_lbl_c = f"Otsu OpenCV (T* = {bm['otsu_cv']['threshold']})"
        else:
            mask_s = result["color_result"]["disease_mask"]
            mask_c = bm["kmeans_cv"]["segmented_img"]
            diff_vis = build_rgb_difference_heatmap(result["kmeans_result"]["segmented_img"], mask_c)
            m_data = km_bm
            t_s = result["kmeans_result"]["exec_time_ms"]
            t_c = bm["kmeans_cv"]["exec_time_ms"]
            t_lbl_s = f"K-means Scratch (k={k_clusters})"
            t_lbl_c = f"K-means OpenCV (k={k_clusters})"

        b1, b2, b3 = st.columns(3)
        with b1:
            with st.container(border=True, key="card_t3_scratch"):
                st.markdown(f'<div class="card-title">🧪 Tự code (NumPy): {t_lbl_s}</div>', unsafe_allow_html=True)
                st.image(result["kmeans_result"]["segmented_img"] if not cmp_target.startswith("Tầng 1") else mask_s, **_STRETCH_KW)
        with b2:
            with st.container(border=True, key="card_t3_opencv"):
                st.markdown(f'<div class="card-title">⚙️ Thư viện OpenCV: {t_lbl_c}</div>', unsafe_allow_html=True)
                st.image(mask_c, **_STRETCH_KW)
        with b3:
            with st.container(border=True, key="card_t3_diff"):
                st.markdown('<div class="card-title">🔥 Bản đồ sai khác (Difference Heatmap)</div>', unsafe_allow_html=True)
                st.image(diff_vis, **_STRETCH_KW)

        ch_col, tbl_col = st.columns([1.45, 1])
        with ch_col:
            with st.container(border=True, key="card_t3_chart"):
                fig = plot_figma_otsu_chart(
                    result["otsu_result"]["histogram"],
                    result["otsu_result"]["variance_curve"],
                    result["otsu_result"]["threshold"],
                    bm["otsu_cv"]["threshold"],
                )
                st.pyplot(fig, **_STRETCH_KW)
                plt.close(fig)
        with tbl_col:
            with st.container(border=True, key="card_t3_table"):
                st.markdown('<div class="card-title">📊 Bảng Tổng hợp Chỉ số Định lượng</div>', unsafe_allow_html=True)
                st.dataframe(render_metrics_summary_df(m_data, t_s, t_c), **_STRETCH_KW, hide_index=True)


# =============================================================================
# CHẾ ĐỘ 2: CHUYÊN SÂU OTSU (SCRATCH VS OPENCV)
# =============================================================================
elif analysis_mode == "Chuyên sâu Otsu (Scratch vs OpenCV)":
    fb_local: List[str] = []
    prep = call_preprocess(result["original_rgb"], blur_kernel=blur_ksize, blur_sigma=blur_sigma, fallback_tracker=fb_local)

    with st.container(border=True, key="card_m2_opts"):
        otsu_ch = st.radio(
            "Kênh đặc trưng đưa vào Otsu:",
            ["Vegetation Map (ExG × S)", "Saturation (HSV-S)", "Grayscale"],
            horizontal=True,
        )
    target_ch = (
        prep["vegetation_map"]
        if otsu_ch.startswith("Vegetation")
        else (prep["saturation"] if otsu_ch.startswith("Saturation") else prep["grayscale"])
    )

    otsu_s = call_otsu(target_ch, fallback_tracker=fb_local)
    otsu_c = run_opencv_otsu(target_ch)
    otsu_m = compute_segmentation_metrics_scratch(otsu_s["binary_mask"], otsu_c["binary_mask"])
    diff_map = build_binary_difference_map(otsu_s["binary_mask"], otsu_c["binary_mask"])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Ngưỡng T* (Scratch)", f"{otsu_s['threshold']}", f"OpenCV T* = {otsu_c['threshold']}")
    with k2:
        render_kpi_card("Độ tương đồng IoU", f"{otsu_m['iou_pct']:.2f}%", f"Dice Score: {otsu_m['dice_pct']:.2f}%")
    with k3:
        render_kpi_card("Pixel Accuracy", f"{otsu_m['pixel_accuracy_pct']:.2f}%", f"Precision: {otsu_m['precision_pct']:.1f}%")
    with k4:
        render_kpi_card("PSNR & Tốc độ", f"{otsu_m['psnr']:.1f} dB", f"Scratch: {otsu_s['exec_time_ms']:.2f} ms | CV: {otsu_c['exec_time_ms']:.2f} ms")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True, key="card_m2_scratch"):
            st.markdown('<div class="card-title">🧪 Mặt nạ Otsu Scratch (NumPy)</div>', unsafe_allow_html=True)
            st.image(otsu_s["binary_mask"], caption=f"T* = {otsu_s['threshold']} ({otsu_s['exec_time_ms']:.2f} ms)", **_STRETCH_KW)
    with c2:
        with st.container(border=True, key="card_m2_opencv"):
            st.markdown('<div class="card-title">⚙️ Mặt nạ Otsu OpenCV</div>', unsafe_allow_html=True)
            st.image(otsu_c["binary_mask"], caption=f"T* = {otsu_c['threshold']} ({otsu_c['exec_time_ms']:.2f} ms)", **_STRETCH_KW)
    with c3:
        with st.container(border=True, key="card_m2_diff"):
            st.markdown('<div class="card-title">🔥 Bản đồ sai khác (Diff Map)</div>', unsafe_allow_html=True)
            st.image(diff_map, caption="Xanh: Khớp | Đỏ: Chỉ Scratch | Lam: Chỉ OpenCV", **_STRETCH_KW)

    ch_col, tbl_col = st.columns([1.45, 1])
    with ch_col:
        with st.container(border=True, key="card_m2_chart"):
            fig = plot_figma_otsu_chart(
                otsu_s["histogram"], otsu_s["variance_curve"], otsu_s["threshold"], otsu_c["threshold"]
            )
            st.pyplot(fig, **_STRETCH_KW)
            plt.close(fig)
    with tbl_col:
        with st.container(border=True, key="card_m2_table"):
            st.markdown('<div class="card-title">📊 Bảng Chỉ số Đối sánh Otsu</div>', unsafe_allow_html=True)
            st.dataframe(render_metrics_summary_df(otsu_m, otsu_s["exec_time_ms"], otsu_c["exec_time_ms"]), **_STRETCH_KW, hide_index=True)


# =============================================================================
# CHẾ ĐỘ 3: CHUYÊN SÂU K-MEANS (SCRATCH VS OPENCV)
# =============================================================================
else:
    fb_local = []
    with st.container(border=True, key="card_m3_opts"):
        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            use_roi = st.checkbox("Chỉ phân cụm trong vùng Leaf ROI (Lấp lỗ BFS)", value=True)
        with opt_c2:
            use_patho = st.checkbox("Dùng không gian 6 kênh nhạy bệnh học", value=True)

    roi_m = result["leaf_mask"] if use_roi else None
    km_in = build_pathology_feature_space(result["blurred"], result["hsv"]) if use_patho else result["blurred"]

    km_s = call_kmeans(km_in, k=k_clusters, max_iter=100, tol=1e-3, random_state=42, roi_mask=roi_m, fallback_tracker=fb_local)
    km_c = run_opencv_kmeans(km_in, k=k_clusters, max_iter=100, tol=1e-3, roi_mask=roi_m)

    seg_s = km_s["segmented_img"][:, :, :3].astype(np.uint8)
    seg_c = km_c["segmented_img"][:, :, :3].astype(np.uint8)
    sim_m = compute_image_similarity_metrics(seg_s, seg_c)
    diff_heat = build_rgb_difference_heatmap(seg_s, seg_c)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Số cụm & Hội tụ", f"K = {k_clusters}", f"Số vòng lặp Scratch: {km_s['n_iter']} iter")
    with k2:
        render_kpi_card("Độ tương đồng IoU", f"{bm['kmeans_disease_metrics']['iou_pct']:.1f}%", f"Dice: {bm['kmeans_disease_metrics']['dice_pct']:.1f}%")
    with k3:
        render_kpi_card("Ảnh phân đoạn PSNR", f"{sim_m['psnr']:.2f} dB", f"MSE: {sim_m['mse']:.4f}")
    with k4:
        render_kpi_card("Thời gian thực thi", f"{km_s['exec_time_ms']:.1f} ms", f"OpenCV: {km_c['exec_time_ms']:.1f} ms")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True, key="card_m3_scratch"):
            st.markdown('<div class="card-title">🧪 K-means Scratch (NumPy)</div>', unsafe_allow_html=True)
            st.image(seg_s, caption=f"{km_s['n_iter']} iter | {km_s['exec_time_ms']:.2f} ms", **_STRETCH_KW)
    with c2:
        with st.container(border=True, key="card_m3_opencv"):
            st.markdown('<div class="card-title">⚙️ K-means OpenCV</div>', unsafe_allow_html=True)
            st.image(seg_c, caption=f"cv2.kmeans | {km_c['exec_time_ms']:.2f} ms", **_STRETCH_KW)
    with c3:
        with st.container(border=True, key="card_m3_diff"):
            st.markdown('<div class="card-title">🔥 Bản đồ nhiệt sai khác</div>', unsafe_allow_html=True)
            st.image(diff_heat, caption="Magma Heatmap (|Scratch - OpenCV|)", **_STRETCH_KW)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        with st.container(border=True, key="card_m3_centroids"):
            st.markdown('<div class="card-title">🎨 Tọa độ Tâm cụm RGB (Scratch Centroids)</div>', unsafe_allow_html=True)
            cent_df = pd.DataFrame(np.round(km_s["centroids"][:, :3], 1), columns=["R", "G", "B"])
            cent_df.insert(0, "Cluster", [f"Cụm {i}" for i in range(len(cent_df))])
            st.dataframe(cent_df, **_STRETCH_KW, hide_index=True)
    with t_col2:
        with st.container(border=True, key="card_m3_table"):
            st.markdown('<div class="card-title">📊 Chỉ số Đối sánh K-means</div>', unsafe_allow_html=True)
            km_table = pd.DataFrame(
                [
                    {"Chỉ số": "MSE (Segmented RGB)", "Giá trị": f"{sim_m['mse']:.4f}"},
                    {"Chỉ số": "PSNR (dB)", "Giá trị": f"{sim_m['psnr']:.2f} dB"},
                    {"Chỉ số": "Disease Mask IoU", "Giá trị": f"{bm['kmeans_disease_metrics']['iou_pct']:.2f}%"},
                    {"Chỉ số": "Disease Mask Dice", "Giá trị": f"{bm['kmeans_disease_metrics']['dice_pct']:.2f}%"},
                    {"Chỉ số": "Thời gian chạy (Scratch / CV)", "Giá trị": f"{km_s['exec_time_ms']:.2f} ms / {km_c['exec_time_ms']:.2f} ms"},
                ]
            )
            st.dataframe(km_table, **_STRETCH_KW, hide_index=True)
