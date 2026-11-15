"""Kiểm thử độ chính xác (Accuracy) module Vai trò 2: data_io, preprocess, otsu.

Khác với `test_contracts.py` (chỉ kiểm tra dtype/shape/keys), bộ test này ĐỐI SÁNH SỐ HỌC
kết quả tự code với OpenCV (OpenCV chỉ được dùng trong test để làm mốc tham chiếu):

- RGB -> Gray: |scratch - cv2.COLOR_RGB2GRAY| <= 1
- RGB -> HSV : so với cv2.COLOR_RGB2HSV_FULL (H thang 0..255), sai khác vòng (circular) <= 1
- Gaussian   : so với cv2.GaussianBlur(BORDER_REFLECT_101) <= 1
- Mean       : so với cv2.blur <= 1
- Otsu       : |T*_scratch - T*_opencv| <= 1 trên ảnh ngẫu nhiên, ảnh 2 mode, ảnh có ROI
- CSV        : ghi rồi đọc lại phải khớp tuyệt đối (round-trip)

Chạy: `python -m pytest tests/test_role2_preprocess_otsu.py -v`
"""

import io
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.data_io import load_csv_matrix, load_image_or_csv, matrix_to_csv_bytes, save_matrix_to_csv
from src.otsu import compute_between_class_variance, compute_histogram_scratch, otsu_threshold_scratch
from src.preprocess import (
    convolve2d_scratch,
    create_gaussian_kernel_2d,
    gaussian_filter_scratch,
    mean_filter_scratch,
    rgb_to_gray_scratch,
    rgb_to_hsv_scratch,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _rand_rgb(seed: int, h: int = 97, w: int = 131) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _all_rgb_colors() -> np.ndarray:
    """Toàn bộ 2^24 màu là quá lớn; lấy lưới 0..255 bước 5 (~140k màu) + các màu biên."""
    v = np.unique(np.r_[np.arange(0, 256, 5), [1, 254, 255]]).astype(np.uint8)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1).reshape(-1, 1, 3)


# =============================================================================
# 1. COLOR SPACE
# =============================================================================

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gray_matches_opencv(seed):
    rgb = _rand_rgb(seed)
    diff = np.abs(rgb_to_gray_scratch(rgb).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(int))
    assert diff.max() <= 1


def test_gray_matches_opencv_color_grid():
    rgb = _all_rgb_colors()
    diff = np.abs(rgb_to_gray_scratch(rgb).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(int))
    assert diff.max() <= 1
    assert np.mean(diff == 0) > 0.95  # gần như trùng khớp tuyệt đối


def test_hsv_matches_opencv_full_scale():
    rgb = _all_rgb_colors()
    ours = rgb_to_hsv_scratch(rgb).astype(int)
    ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(int)
    dh = np.abs(ours[..., 0] - ref[..., 0])
    dh = np.minimum(dh, 256 - dh)  # Hue là đại lượng vòng
    assert dh.max() <= 1
    assert np.abs(ours[..., 1] - ref[..., 1]).max() <= 1
    assert np.abs(ours[..., 2] - ref[..., 2]).max() == 0


def test_hsv_known_colors():
    colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]],
                      dtype=np.uint8)
    hsv = rgb_to_hsv_scratch(colors)[0]
    assert hsv[0].tolist() == [0, 255, 255]      # đỏ: 0°
    assert hsv[1].tolist() == [85, 255, 255]     # lục: 120° -> 85
    assert hsv[2].tolist() == [170, 255, 255]    # lam: 240° -> 170
    assert hsv[3].tolist() == [0, 0, 255]        # trắng: S = 0
    assert hsv[4].tolist() == [0, 0, 0]          # đen: V = 0, không chia cho 0
    assert hsv[5].tolist() == [0, 0, 128]        # xám


@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)])
def test_color_conversion_rejects_bad_shape(bad):
    with pytest.raises(ValueError):
        rgb_to_gray_scratch(bad)
    with pytest.raises(ValueError):
        rgb_to_hsv_scratch(bad)


# =============================================================================
# 2. FILTERS
# =============================================================================

def test_gaussian_kernel_properties():
    k = create_gaussian_kernel_2d(7, 1.5)
    assert k.shape == (7, 7) and k.dtype == np.float64
    assert np.isclose(k.sum(), 1.0)
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])  # đối xứng
    assert np.unravel_index(np.argmax(k), k.shape) == (3, 3)      # cực đại tại tâm
    ref = cv2.getGaussianKernel(7, 1.5)
    assert np.allclose(k, ref @ ref.T, atol=1e-12)


@pytest.mark.parametrize("ksize,sigma", [(0, 1.0), (4, 1.0), (-3, 1.0), (5, 0.0), (5, -1.0)])
def test_gaussian_kernel_invalid_params(ksize, sigma):
    with pytest.raises(ValueError):
        create_gaussian_kernel_2d(ksize, sigma)


@pytest.mark.parametrize("ksize,sigma", [(3, 0.8), (5, 1.0), (7, 2.0), (11, 3.0)])
def test_gaussian_filter_matches_opencv(ksize, sigma):
    rgb = _rand_rgb(10 + ksize)
    ours = gaussian_filter_scratch(rgb, kernel_size=ksize, sigma=sigma)
    ref = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    assert ours.shape == rgb.shape and ours.dtype == np.uint8
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_mean_filter_matches_opencv(ksize):
    gray = _rand_rgb(20)[:, :, 0]
    ours = mean_filter_scratch(gray, kernel_size=ksize)
    ref = cv2.blur(gray, (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_separable_equals_direct_2d_convolution():
    rgb = _rand_rgb(30)
    kernel = create_gaussian_kernel_2d(5, 1.2)
    direct = convolve2d_scratch(rgb, kernel)
    separable = gaussian_filter_scratch(rgb, 5, 1.2)
    assert np.abs(direct.astype(int) - separable.astype(int)).max() <= 1


def test_convolution_is_true_convolution_not_correlation():
    img = np.zeros((7, 7), np.uint8)
    img[3, 3] = 100
    kernel = np.zeros((3, 3))
    kernel[0, 0] = 1.0  # trọng số tại offset (-1, -1)
    out = convolve2d_scratch(img, kernel, padding_mode="constant")
    # Tích chập xung đơn vị "đóng dấu" kernel NGUYÊN BẢN quanh xung -> đỉnh tại (2, 2).
    # (Tương quan/correlation sẽ cho kernel bị lật -> đỉnh tại (4, 4).)
    assert out[2, 2] == 100 and int(out.sum()) == 100
    ref = cv2.filter2D(img, -1, kernel[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
    assert np.array_equal(out, ref)


def test_filters_preserve_constant_image_and_tiny_inputs():
    flat = np.full((20, 20, 3), 77, np.uint8)
    assert np.all(gaussian_filter_scratch(flat, 5, 1.0) == 77)
    assert np.all(mean_filter_scratch(flat, 5) == 77)
    tiny = np.full((1, 1, 3), 9, np.uint8)
    assert gaussian_filter_scratch(tiny, 5, 1.0).shape == (1, 1, 3)


def test_filters_reduce_noise_variance():
    rng = np.random.default_rng(5)
    noisy = np.clip(128 + rng.normal(0, 25, (128, 128)), 0, 255).astype(np.uint8)
    assert gaussian_filter_scratch(noisy, 5, 1.0).std() < 0.5 * noisy.std()
    assert mean_filter_scratch(noisy, 5).std() < 0.5 * noisy.std()


# =============================================================================
# 3. OTSU
# =============================================================================

def _cv_otsu(gray: np.ndarray) -> int:
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(round(t))


def test_histogram_matches_opencv_calchist():
    gray = _rand_rgb(40)[:, :, 1]
    ours = compute_histogram_scratch(gray)
    ref = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    assert ours.dtype == np.int64 and np.array_equal(ours, ref)


def test_histogram_roi_counts_only_roi():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    roi = np.zeros((10, 10), np.uint8)
    roi[:2] = 255
    hist = compute_histogram_scratch(gray, roi)
    assert hist.sum() == 20 and np.all(hist[:20] == 1) and hist[20:].sum() == 0
    with pytest.raises(ValueError):
        compute_histogram_scratch(gray, roi[:5])


@pytest.mark.parametrize("seed", range(40))
def test_otsu_threshold_within_one_of_opencv_random_mixtures(seed):
    """Ảnh hỗn hợp 2–3 phân phối Gauss ngẫu nhiên (giống histogram lá/nền thực tế)."""
    rng = np.random.default_rng(seed)
    n_modes = rng.integers(2, 4)
    sizes = rng.dirichlet(np.ones(n_modes)) * 20000
    parts = [rng.normal(rng.uniform(20, 235), rng.uniform(3, 40), int(s) + 1) for s in sizes]
    gray = np.clip(np.concatenate(parts), 0, 255).astype(np.uint8)
    gray = gray[: (gray.size // 100) * 100].reshape(100, -1)
    res = otsu_threshold_scratch(gray)
    assert abs(res["threshold"] - _cv_otsu(gray)) <= 1


@pytest.mark.parametrize("lo,hi", [(50, 200), (0, 255), (10, 11), (100, 180)])
def test_otsu_two_level_plateau_matches_opencv(lo, hi):
    """Histogram chỉ có 2 mức -> σ²_B phẳng trên cả đoạn [lo, hi): phải chọn t nhỏ nhất như OpenCV."""
    gray = np.full((40, 40), lo, np.uint8)
    gray[:, 25:] = hi
    res = otsu_threshold_scratch(gray)
    assert res["threshold"] == _cv_otsu(gray) == lo
    assert np.array_equal(res["binary_mask"], np.where(gray > lo, 255, 0).astype(np.uint8))


def test_otsu_mask_matches_opencv_exactly_when_threshold_equal():
    gray = cv2.GaussianBlur(_rand_rgb(50, 128, 128)[:, :, 0], (9, 9), 3)
    res = otsu_threshold_scratch(gray)
    t_cv, mask_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert res["threshold"] == int(t_cv)
    assert np.array_equal(res["binary_mask"], mask_cv)


def test_otsu_variance_curve_peak_is_threshold():
    gray = _rand_rgb(60)[:, :, 2]
    res = otsu_threshold_scratch(gray)
    curve = res["variance_curve"]
    assert curve.shape == (256,) and curve.dtype == np.float64 and np.all(curve >= 0)
    assert np.isclose(curve[res["threshold"]], curve.max())
    assert np.allclose(curve, compute_between_class_variance(res["histogram"]))


def test_otsu_variance_formula_matches_definition():
    """σ²_B(t) theo công thức rút gọn == ω0·ω1·(μ0 − μ1)² theo định nghĩa."""
    gray = _rand_rgb(61)[:, :, 0]
    hist = compute_histogram_scratch(gray).astype(float)
    p = hist / hist.sum()
    i = np.arange(256)
    expected = np.zeros(256)
    for t in range(256):
        w0, w1 = p[: t + 1].sum(), p[t + 1:].sum()
        if w0 > 0 and w1 > 0:
            mu0 = (i[: t + 1] * p[: t + 1]).sum() / w0
            mu1 = (i[t + 1:] * p[t + 1:]).sum() / w1
            expected[t] = w0 * w1 * (mu0 - mu1) ** 2
    assert np.allclose(compute_between_class_variance(hist.astype(np.int64)), expected)


def test_otsu_roi_matches_opencv_on_roi_pixels():
    rng = np.random.default_rng(70)
    gray = np.clip(rng.normal(90, 30, (120, 120)), 0, 255).astype(np.uint8)
    roi = np.zeros_like(gray)
    roi[30:90, 30:90] = 255
    gray[roi > 0] = np.clip(gray[roi > 0].astype(int) + 80 * (rng.random(3600) > 0.6), 0, 255)
    res = otsu_threshold_scratch(gray, roi_mask=roi)
    assert abs(res["threshold"] - _cv_otsu(gray[roi > 0].reshape(-1, 1))) <= 1
    assert not np.any(res["binary_mask"][roi == 0])


def test_otsu_uniform_and_empty_roi_do_not_crash():
    flat = np.full((16, 16), 120, np.uint8)
    res = otsu_threshold_scratch(flat)
    assert res["threshold"] == _cv_otsu(flat) == 0
    res_empty = otsu_threshold_scratch(flat, roi_mask=np.zeros_like(flat))
    assert res_empty["threshold"] == 0 and not np.any(res_empty["binary_mask"])


def test_otsu_invert_flag():
    gray = np.full((10, 10), 30, np.uint8)
    gray[:, 5:] = 220
    normal = otsu_threshold_scratch(gray)["binary_mask"]
    inverted = otsu_threshold_scratch(gray, invert=True)["binary_mask"]
    assert np.array_equal(inverted, 255 - normal)


def test_otsu_separability_range():
    two_mode = np.zeros((10, 10), np.uint8)
    two_mode[:, 5:] = 255
    assert otsu_threshold_scratch(two_mode)["separability"] == pytest.approx(1.0)
    noise = _rand_rgb(80)[:, :, 0]
    assert 0.0 < otsu_threshold_scratch(noise)["separability"] < 1.0


@pytest.mark.skipif(not any(RAW_DIR.glob("*.jpg")), reason="Chưa có dataset trong data/raw/")
def test_otsu_within_one_of_opencv_on_dataset():
    """Đối sánh trên toàn bộ ảnh thật trong data/raw/ (kênh Gray và kênh S của HSV)."""
    images = sorted(RAW_DIR.glob("*.jpg"))
    for path in images:
        rgb = load_image_or_csv(path)
        blurred = gaussian_filter_scratch(rgb, 5, 1.0)
        for channel in (rgb_to_gray_scratch(blurred), rgb_to_hsv_scratch(blurred)[:, :, 1]):
            t_ours = otsu_threshold_scratch(channel)["threshold"]
            assert abs(t_ours - _cv_otsu(channel)) <= 1, path.name


# =============================================================================
# 4. DATA I/O
# =============================================================================

def test_csv_round_trip_exact(tmp_path):
    mat = _rand_rgb(90, 33, 47)[:, :, 0]
    out = tmp_path / "sub" / "m.csv"
    save_matrix_to_csv(mat, out)
    back = load_image_or_csv(out)
    assert back.dtype == np.uint8 and np.array_equal(back, mat)
    assert matrix_to_csv_bytes(mat) == out.read_bytes()


def test_csv_reader_accepts_header_semicolon_and_buffer():
    text = "c0;c1;c2\n1;2;3\n\n4;5;6\n"
    buf = io.BytesIO(text.encode())
    buf.name = "x.csv"
    assert load_image_or_csv(buf).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert load_csv_matrix(b"7 8\n9 10\n").tolist() == [[7, 8], [9, 10]]


def test_csv_reader_rejects_ragged_and_warns_out_of_range(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("1,2,3\n4,5\n")
    with pytest.raises(ValueError):
        load_image_or_csv(bad)
    oor = tmp_path / "oor.csv"
    oor.write_text("-5,300\n10.6,20\n")
    with pytest.warns(RuntimeWarning):
        assert load_image_or_csv(oor).tolist() == [[0, 255], [11, 20]]


def test_save_matrix_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError):
        save_matrix_to_csv(np.zeros((2, 2, 3)), tmp_path / "x.csv")
    with pytest.raises(ValueError):
        matrix_to_csv_bytes(np.zeros(5))


def test_load_image_rgb_order_resize_and_alpha(tmp_path):
    from PIL import Image

    rgb = np.zeros((600, 300, 3), np.uint8)
    rgb[..., 0] = 200  # kênh đỏ
    path = tmp_path / "red.png"
    Image.fromarray(rgb).save(path)
    img = load_image_or_csv(path, max_size=512)
    assert img.shape == (512, 256, 3) and img.dtype == np.uint8
    assert img[..., 0].mean() == 200 and img[..., 2].max() == 0  # RGB, không phải BGR

    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba_path = tmp_path / "transparent.png"
    Image.fromarray(rgba).save(rgba_path)
    assert np.all(load_image_or_csv(rgba_path) == 255)  # nền trong suốt -> trắng

    with open(path, "rb") as fh:
        assert load_image_or_csv(fh.read(), max_size=0).shape == (600, 300, 3)  # bytes không tên


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_or_csv(tmp_path / "missing.png")
    bad = tmp_path / "x.gif2"
    bad.write_bytes(b"abc")
    with pytest.raises(ValueError):
        load_image_or_csv(bad)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image_or_csv(corrupt)
