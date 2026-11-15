"""Thực nghiệm Vai trò 2: Phân tích Histogram đa kênh & đánh giá tách nền Otsu trên bộ dữ liệu.

Sinh ra (mặc định trong `experiments/results/otsu/`):
- `histogram_by_category.png`   : Histogram R,G,B / H,S,V / Gray (+ ngưỡng T*) cho ảnh đại diện mỗi nhóm.
- `histogram_pixel_classes.png` : Phân bố từng kênh màu của 3 lớp pixel (nền / mô lá khỏe / vết bệnh)
                                  gộp trên toàn bộ dataset, dùng mặt nạ Ground Truth -> chọn kênh tách tốt nhất.
- `otsu_segmentation_grid.png`  : Kết quả tách nền Otsu (Tầng 1) cho cả 30 ảnh, tô màu TP/FP/FN so với GT.
- `otsu_variance_curve.png`     : Ví dụ histogram + đường cong σ²_B(t) và ngưỡng T* (scratch vs OpenCV).
- `otsu_threshold_benchmark.csv`: Bảng T* scratch vs OpenCV, |ΔT|, thời gian, IoU/Dice so với GT cho mọi ảnh x kênh.
- `lesion_otsu_benchmark.csv`   : Otsu phân ngưỡng đốm bệnh trong ROI lá (ứng dụng thứ 2 của Otsu).
- `speed_benchmark.csv`         : Tốc độ hàm tự code vs OpenCV (Gray, HSV, Gaussian, Mean, Otsu).
- `summary.md`                  : Bảng tổng hợp và nhận xét.

OpenCV CHỈ được dùng ở đây làm mốc đối sánh (baseline), không dùng trong module thuật toán.

Chạy từ thư mục gốc repository:
    python -m experiments.otsu_analysis
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data_io import load_image_or_csv
from src.otsu import compute_histogram_scratch, otsu_threshold_scratch
from src.preprocess import gaussian_filter_scratch, mean_filter_scratch, rgb_to_gray_scratch, rgb_to_hsv_scratch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DEFAULT = ROOT / "experiments" / "results" / "otsu"

CATEGORY_ORDER = ["healthy", "mild", "moderate", "severe", "critical", "challenge"]
CAT_VI = {"healthy": "Khỏe", "mild": "Nhẹ", "moderate": "Trung bình", "severe": "Nặng",
          "critical": "Nghiêm trọng", "challenge": "Điều kiện khó"}


# =============================================================================
# KÊNH ĐẶC TRƯNG CHO TÁCH NỀN
# =============================================================================

def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Chỉ số ExG = 2g − r − b trên tọa độ màu chuẩn hóa (r = R/(R+G+B)...), co về [0, 255]."""
    f = rgb.astype(np.float64)
    s = f.sum(axis=2) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    exg = 2 * g - r - b  # ∈ [-1, 2]
    return np.clip(np.round((exg + 1) / 3 * 255), 0, 255).astype(np.uint8)


def auto_polarity_otsu(channel: np.ndarray) -> Dict:
    """Otsu + tự chọn cực tính: lớp nào chiếm phần lớn VIỀN ẢNH thì là nền.

    Cần thiết với kênh Gray: lá tối hơn nền giấy trắng nhưng sáng hơn nền tối.
    """
    res = otsu_threshold_scratch(channel)
    mask = res["binary_mask"] > 0
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    inverted = border.mean() > 0.5
    if inverted:
        res = otsu_threshold_scratch(channel, invert=True)
    res["inverted"] = bool(inverted)
    return res


def cv_otsu_threshold(channel: np.ndarray, roi: Optional[np.ndarray] = None):
    if cv2 is None:
        return None, None
    data = channel if roi is None else channel[roi].reshape(-1, 1)
    t0 = time.perf_counter()
    t, _ = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(round(t)), (time.perf_counter() - t0) * 1000


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {"iou": float(iou), "dice": float(dice)}


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Giữ thành phần liên thông lớn nhất (BFS 4-lân cận, dùng riêng cho thực nghiệm)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    best: List = []
    flat = mask.ravel()
    for start in np.flatnonzero(flat):
        if seen.flat[start]:
            continue
        comp = [start]
        seen.flat[start] = True
        i = 0
        while i < len(comp):
            p = comp[i]
            i += 1
            y, x = divmod(p, w)
            for q, ok in ((p - w, y > 0), (p + w, y < h - 1), (p - 1, x > 0), (p + 1, x < w - 1)):
                if ok and flat[q] and not seen.flat[q]:
                    seen.flat[q] = True
                    comp.append(q)
        if len(comp) > len(best):
            best = comp
    out = np.zeros(h * w, bool)
    out[best] = True
    return out.reshape(h, w)


def load_gt(name: str):
    stem = Path(name).stem
    leaf = np.asarray(Image.open(RAW / "ground_truth" / f"{stem}_leaf.png")) > 0
    lesion = np.asarray(Image.open(RAW / "ground_truth" / f"{stem}_lesion.png")) > 0
    return leaf, lesion


def read_metadata() -> List[Dict]:
    with open(RAW / "metadata.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# =============================================================================
# THỰC NGHIỆM
# =============================================================================

def run_leaf_segmentation(meta: List[Dict]) -> (List[Dict], Dict):
    rows, cache = [], {}
    try:
        from src.pipeline import compute_vegetation_feature_map  # kênh do nhóm trưởng thiết kế
    except Exception:  # pragma: no cover
        compute_vegetation_feature_map = None

    for m in meta:
        name = m["filename"]
        rgb = load_image_or_csv(RAW / name, max_size=0)
        leaf_gt, lesion_gt = load_gt(name)
        blurred = gaussian_filter_scratch(rgb, kernel_size=5, sigma=1.0)
        gray = rgb_to_gray_scratch(blurred)
        hsv = rgb_to_hsv_scratch(blurred)
        channels = {"Gray": gray, "S (HSV)": hsv[..., 1], "ExG": excess_green(blurred)}
        if compute_vegetation_feature_map is not None:
            channels["VegMap (pipeline)"] = compute_vegetation_feature_map(blurred, hsv)
        cache[name] = {"rgb": rgb, "gray": gray, "hsv": hsv, "leaf_gt": leaf_gt, "lesion_gt": lesion_gt,
                       "channels": channels, "meta": m}
        for ch_name, ch in channels.items():
            res = auto_polarity_otsu(ch)
            pred = res["binary_mask"] > 0
            pred_lcc = largest_component(pred)
            t_cv, ms_cv = cv_otsu_threshold(ch)
            met = seg_metrics(pred, leaf_gt)
            met_lcc = seg_metrics(pred_lcc, leaf_gt)
            rows.append({
                "filename": name, "category": m["category"], "condition": m["condition"],
                "background": m["background"], "channel": ch_name,
                "T_scratch": res["threshold"], "T_opencv": t_cv,
                "abs_dT": None if t_cv is None else abs(res["threshold"] - t_cv),
                "inverted": res["inverted"], "separability": round(res["separability"], 4),
                "time_scratch_ms": round(res["exec_time_ms"], 3),
                "time_opencv_ms": None if ms_cv is None else round(ms_cv, 3),
                "iou_raw": round(met["iou"], 4), "dice_raw": round(met["dice"], 4),
                "iou_lcc": round(met_lcc["iou"], 4), "dice_lcc": round(met_lcc["dice"], 4),
            })
            cache[name].setdefault("pred", {})[ch_name] = (pred_lcc, res)
    return rows, cache


def run_lesion_otsu(cache: Dict) -> List[Dict]:
    """Otsu trong ROI = phiến lá GT trên kênh H (Hue): đốm vàng/nâu có Hue thấp hơn mô xanh."""
    rows = []
    for name, c in cache.items():
        roi = c["leaf_gt"]
        hue = c["hsv"][..., 0]
        res = otsu_threshold_scratch(hue, roi_mask=roi, invert=True)  # lesion: H <= T*
        pred = res["binary_mask"] > 0
        t_cv, _ = cv_otsu_threshold(hue, roi)
        gt = c["lesion_gt"]
        met = seg_metrics(pred, gt) if gt.any() else {"iou": float("nan"), "dice": float("nan")}
        ratio_pred = pred.sum() / roi.sum() * 100
        ratio_gt = gt.sum() / roi.sum() * 100
        rows.append({
            "filename": name, "category": c["meta"]["category"], "condition": c["meta"]["condition"],
            "T_scratch": res["threshold"], "T_opencv": t_cv,
            "abs_dT": None if t_cv is None else abs(res["threshold"] - t_cv),
            "separability": round(res["separability"], 4),
            "gt_ratio_pct": round(ratio_gt, 2), "otsu_ratio_pct": round(ratio_pred, 2),
            "iou": round(met["iou"], 4), "dice": round(met["dice"], 4),
        })
        c["lesion_pred"] = pred
    return rows


def run_speed_benchmark(cache: Dict, repeats: int = 5) -> List[Dict]:
    if cv2 is None:
        return []
    rgb = next(iter(cache.values()))["rgb"]
    rgb = np.asarray(Image.fromarray(rgb).resize((512, 512)))
    gray = rgb_to_gray_scratch(rgb)

    def timeit(fn):
        fn()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    cases = [
        ("RGB→Gray", lambda: rgb_to_gray_scratch(rgb), lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
         lambda: np.abs(rgb_to_gray_scratch(rgb).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).max()),
        ("RGB→HSV", lambda: rgb_to_hsv_scratch(rgb), lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL), None),
        ("Gaussian 5x5 (RGB)", lambda: gaussian_filter_scratch(rgb, 5, 1.0),
         lambda: cv2.GaussianBlur(rgb, (5, 5), 1.0),
         lambda: np.abs(gaussian_filter_scratch(rgb, 5, 1.0).astype(int) - cv2.GaussianBlur(rgb, (5, 5), 1.0)).max()),
        ("Mean 5x5 (Gray)", lambda: mean_filter_scratch(gray, 5), lambda: cv2.blur(gray, (5, 5)),
         lambda: np.abs(mean_filter_scratch(gray, 5).astype(int) - cv2.blur(gray, (5, 5))).max()),
        ("Otsu (Gray)", lambda: otsu_threshold_scratch(gray),
         lambda: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU),
         lambda: abs(otsu_threshold_scratch(gray)["threshold"]
                     - int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))),
    ]
    rows = []
    for name, ours, ref, diff in cases:
        t_ours, t_ref = timeit(ours), timeit(ref)
        rows.append({"operation": name, "image": "512x512", "scratch_ms": round(t_ours, 3),
                     "opencv_ms": round(t_ref, 3), "ratio": round(t_ours / max(t_ref, 1e-6), 1),
                     "max_abs_diff": "" if diff is None else int(diff())})
    return rows


# =============================================================================
# BIỂU ĐỒ
# =============================================================================

def plot_histograms_by_category(cache: Dict, out: Path):
    picks = ["healthy_01.jpg", "mild_01.jpg", "moderate_01.jpg", "severe_01.jpg", "critical_01.jpg",
             "shadow_01.jpg", "glare_01.jpg", "noise_01.jpg", "yellowvein_01.jpg", "lowlight_01.jpg"]
    picks = [p for p in picks if p in cache]
    fig, axes = plt.subplots(len(picks), 4, figsize=(17, 2.6 * len(picks)),
                             gridspec_kw={"width_ratios": [1.0, 1.4, 1.4, 1.4]})
    x = np.arange(256)
    for r, name in enumerate(picks):
        c = cache[name]
        ax = axes[r, 0]
        ax.imshow(c["rgb"])
        ax.set_title(f"{Path(name).stem}  (R={float(c['meta']['gt_disease_ratio_pct']):.1f}%)", fontsize=9)
        ax.axis("off")
        for ch, col in zip(range(3), ("#d62728", "#2ca02c", "#1f77b4")):
            axes[r, 1].plot(x, compute_histogram_scratch(c["rgb"][..., ch]), color=col, lw=1)
        axes[r, 1].set_title("Histogram R, G, B", fontsize=9)
        for ch, col, lab in zip(range(3), ("#9467bd", "#ff7f0e", "#7f7f7f"), ("H", "S", "V")):
            axes[r, 2].plot(x, compute_histogram_scratch(c["hsv"][..., ch]), color=col, lw=1, label=lab)
        axes[r, 2].set_title("Histogram H, S, V (thang 0–255)", fontsize=9)
        axes[r, 2].legend(fontsize=7, loc="upper right")
        res = otsu_threshold_scratch(c["gray"])
        axes[r, 3].bar(x, res["histogram"], width=1.0, color="#555555")
        axes[r, 3].axvline(res["threshold"], color="#d62728", lw=1.5, label=f"T* = {res['threshold']}")
        axes[r, 3].set_title(f"Histogram Gray + ngưỡng Otsu (η = {res['separability']:.2f})", fontsize=9)
        axes[r, 3].legend(fontsize=7, loc="upper right")
        for k in (1, 2, 3):
            axes[r, k].set_xlim(0, 255)
            axes[r, k].tick_params(labelsize=7)
            axes[r, k].set_yticks([])
    fig.suptitle("Phân bố Histogram đa kênh màu theo nhóm ảnh (sau lọc Gauss 5x5, σ=1)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "histogram_by_category.png", dpi=90)
    plt.close(fig)


def plot_pixel_class_histograms(cache: Dict, out: Path):
    """Phân bố kênh màu của 3 lớp pixel (dùng GT) — trả lời: kênh nào tách lá/nền, mô khỏe/bệnh tốt nhất."""
    names = [n for n, c in cache.items() if c["meta"]["condition"] == "normal"]
    chans = {"R": lambda c: c["rgb"][..., 0], "G": lambda c: c["rgb"][..., 1], "B": lambda c: c["rgb"][..., 2],
             "Gray": lambda c: c["gray"], "H": lambda c: c["hsv"][..., 0], "S": lambda c: c["hsv"][..., 1],
             "V": lambda c: c["hsv"][..., 2], "ExG": lambda c: c["channels"]["ExG"]}
    classes = {"Nền": "#8c8c8c", "Mô lá khỏe": "#2ca02c", "Vết bệnh (GT)": "#b8860b"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    summary = {}
    for ax, (ch_name, getter) in zip(axes.ravel(), chans.items()):
        hists = {k: np.zeros(256) for k in classes}
        for n in names:
            c = cache[n]
            ch = getter(c)
            leaf, les = c["leaf_gt"], c["lesion_gt"]
            hists["Nền"] += compute_histogram_scratch(ch, ~leaf)
            hists["Mô lá khỏe"] += compute_histogram_scratch(ch, leaf & ~les)
            hists["Vết bệnh (GT)"] += compute_histogram_scratch(ch, les)
        for k, col in classes.items():
            p = hists[k] / max(hists[k].sum(), 1)
            ax.fill_between(np.arange(256), p, color=col, alpha=0.45, label=k, step="mid")
        ax.set_title(ch_name, fontsize=11)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        # độ chồng lấn (overlap) = Σ min(p, q): càng nhỏ càng tách tốt
        pn = {k: v / max(v.sum(), 1) for k, v in hists.items()}
        summary[ch_name] = {
            "overlap_bg_leaf": float(np.minimum(pn["Nền"], pn["Mô lá khỏe"] + 0).sum()),
            "overlap_healthy_lesion": float(np.minimum(pn["Mô lá khỏe"], pn["Vết bệnh (GT)"]).sum()),
        }
        ax.text(0.02, 0.95, f"chồng lấn nền–lá: {summary[ch_name]['overlap_bg_leaf']:.2f}\n"
                            f"chồng lấn khỏe–bệnh: {summary[ch_name]['overlap_healthy_lesion']:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Phân bố mức kênh màu theo lớp pixel Ground Truth ({len(names)} ảnh điều kiện thường, "
                 "chuẩn hóa từng lớp)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "histogram_pixel_classes.png", dpi=90)
    plt.close(fig)
    return summary


def plot_segmentation_grid(cache: Dict, channel: str, out: Path, rows_tbl: List[Dict]):
    names = list(cache.keys())
    ncol = 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow))
    iou_lookup = {(r["filename"], r["channel"]): r["iou_lcc"] for r in rows_tbl}
    for ax, name in zip(axes.ravel(), names):
        c = cache[name]
        pred, res = c["pred"][channel]
        gt = c["leaf_gt"]
        vis = c["rgb"].astype(np.float64) * 0.55
        vis[pred & gt] = vis[pred & gt] * 0.4 + np.array([0, 200, 0]) * 0.6
        vis[pred & ~gt] = np.array([230, 30, 30])
        vis[~pred & gt] = np.array([30, 80, 255])
        ax.imshow(vis.astype(np.uint8))
        ax.set_title(f"{Path(name).stem}\nT*={res['threshold']}  IoU={iou_lookup[(name, channel)]:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Tầng 1 — Tách nền lá bằng Otsu tự code trên kênh '{channel}' "
                 "(xanh lá = đúng, đỏ = nhận nhầm nền thành lá, xanh dương = sót lá)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    safe = channel.split(" ")[0].lower()
    fig.savefig(out / f"otsu_segmentation_grid_{safe}.png", dpi=80)
    plt.close(fig)


def plot_variance_curve(cache: Dict, out: Path):
    name = "moderate_01.jpg" if "moderate_01.jpg" in cache else next(iter(cache))
    c = cache[name]
    ch = c["channels"]["S (HSV)"]
    res = otsu_threshold_scratch(ch)
    t_cv, _ = cv_otsu_threshold(ch)
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax1.bar(np.arange(256), res["histogram"], width=1.0, color="#4c9a6a", alpha=0.8, label="Histogram kênh S")
    ax1.set_xlabel("Mức giá trị t (0–255)")
    ax1.set_ylabel("Số pixel")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(256), res["variance_curve"], color="#d62728", lw=2, label="σ²_B(t)")
    ax2.set_ylabel("Phương sai giữa hai lớp σ²_B(t)")
    ax2.axvline(res["threshold"], color="#d62728", ls="--", label=f"T* tự code = {res['threshold']}")
    if t_cv is not None:
        ax2.axvline(t_cv, color="black", ls=":", label=f"T* OpenCV = {t_cv}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax1.set_title(f"{Path(name).stem}: T* = argmax σ²_B(t),  η = σ²_B(T*)/σ²_T = {res['separability']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "otsu_variance_curve.png", dpi=100)
    plt.close(fig)


def plot_lesion_examples(cache: Dict, out: Path, lesion_rows: List[Dict]):
    picks = [n for n in ("mild_01.jpg", "moderate_01.jpg", "severe_01.jpg", "glare_02.jpg",
                         "yellowvein_01.jpg", "lowlight_01.jpg") if n in cache]
    lookup = {r["filename"]: r for r in lesion_rows}
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6))
    for k, name in enumerate(picks):
        c = cache[name]
        axes[0, k].imshow(c["rgb"])
        axes[0, k].set_title(f"{Path(name).stem}\nGT R = {lookup[name]['gt_ratio_pct']}%", fontsize=9)
        pred, gt = c["lesion_pred"], c["lesion_gt"]
        vis = c["rgb"].astype(np.float64) * 0.5
        vis[pred & gt] = [0, 200, 0]
        vis[pred & ~gt] = [230, 30, 30]
        vis[~pred & gt] = [30, 80, 255]
        axes[1, k].imshow(vis.astype(np.uint8))
        iou = lookup[name]["iou"]
        iou_txt = "—" if np.isnan(iou) else f"{iou:.2f}"
        axes[1, k].set_title(f"Otsu(H) R = {lookup[name]['otsu_ratio_pct']}%  IoU = {iou_txt}", fontsize=9)
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle("Otsu phân ngưỡng đốm bệnh trong ROI phiến lá (kênh Hue) — "
                 "xanh lá = đúng, đỏ = báo động giả, xanh dương = bỏ sót", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "lesion_otsu_examples.png", dpi=85)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def write_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(seg_rows, lesion_rows, speed_rows, overlap, out: Path):
    lines = ["# Kết quả thực nghiệm Vai trò 2 — Histogram & Otsu", ""]
    lines += ["Tự sinh bởi `python -m experiments.otsu_analysis`. Dataset: `data/raw/` "
              f"({len({r['filename'] for r in seg_rows})} ảnh có Ground Truth).", ""]

    lines += ["## 1. Độ lệch ngưỡng T* tự code so với OpenCV", ""]
    all_dt = [r["abs_dT"] for r in seg_rows + lesion_rows if r["abs_dT"] is not None]
    if all_dt:
        lines += [f"- Số lần so sánh: **{len(all_dt)}** (30 ảnh × 4 kênh tách nền + 30 lần Otsu trong ROI lá)",
                  f"- |ΔT| lớn nhất: **{max(all_dt)}**, trung bình: {np.mean(all_dt):.3f}, "
                  f"số lần trùng khớp tuyệt đối: {sum(d == 0 for d in all_dt)}/{len(all_dt)}",
                  f"- Yêu cầu |ΔT| ≤ 1: **{'ĐẠT' if max(all_dt) <= 1 else 'KHÔNG ĐẠT'}**", ""]

    lines += ["## 2. Tách nền lá (Tầng 1) — IoU trung bình so với GT", "",
              "Sau Otsu chỉ giữ thành phần liên thông lớn nhất (LCC). Cột `raw` là mặt nạ Otsu thô.", "",
              "| Kênh | IoU raw | IoU LCC | Dice LCC | IoU LCC thấp nhất (ảnh) |", "|---|---|---|---|---|"]
    for ch in dict.fromkeys(r["channel"] for r in seg_rows):
        rs = [r for r in seg_rows if r["channel"] == ch]
        worst = min(rs, key=lambda r: r["iou_lcc"])
        lines.append(f"| {ch} | {np.mean([r['iou_raw'] for r in rs]):.3f} | {np.mean([r['iou_lcc'] for r in rs]):.3f} "
                     f"| {np.mean([r['dice_lcc'] for r in rs]):.3f} | {worst['iou_lcc']:.3f} ({worst['filename']}) |")
    lines += ["", "IoU LCC theo nền (kênh tốt nhất mỗi nền in đậm):", "",
              "| Nền | " + " | ".join(dict.fromkeys(r["channel"] for r in seg_rows)) + " |",
              "|---|" + "---|" * len(set(r["channel"] for r in seg_rows))]
    chans = list(dict.fromkeys(r["channel"] for r in seg_rows))
    for bg in sorted({r["background"] for r in seg_rows}):
        vals = [np.mean([r["iou_lcc"] for r in seg_rows if r["background"] == bg and r["channel"] == ch])
                for ch in chans]
        best = int(np.argmax(vals))
        lines.append(f"| {bg} | " + " | ".join(f"**{v:.3f}**" if i == best else f"{v:.3f}"
                                               for i, v in enumerate(vals)) + " |")
    lines += ["", "IoU LCC theo điều kiện chụp:", "",
              "| Điều kiện | " + " | ".join(chans) + " |", "|---|" + "---|" * len(chans)]
    for cond in dict.fromkeys(r["condition"] for r in seg_rows):
        vals = [np.mean([r["iou_lcc"] for r in seg_rows if r["condition"] == cond and r["channel"] == ch])
                for ch in chans]
        lines.append(f"| {cond} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    lines += ["", "## 3. Otsu phân ngưỡng đốm bệnh trong ROI lá (kênh Hue)", "",
              "| Nhóm | IoU TB | R% GT TB | R% Otsu TB |", "|---|---|---|---|"]
    for cat in CATEGORY_ORDER:
        rs = [r for r in lesion_rows if r["category"] == cat]
        if not rs:
            continue
        ious = [r["iou"] for r in rs if not np.isnan(r["iou"])]
        lines.append(f"| {CAT_VI[cat]} | {np.mean(ious):.3f} | {np.mean([r['gt_ratio_pct'] for r in rs]):.2f} | "
                     f"{np.mean([r['otsu_ratio_pct'] for r in rs]):.2f} |" if ious else
                     f"| {CAT_VI[cat]} | — (không có vết bệnh) | 0.00 | "
                     f"{np.mean([r['otsu_ratio_pct'] for r in rs]):.2f} |")

    lines += ["", "## 4. Độ chồng lấn histogram giữa các lớp pixel (càng nhỏ càng dễ phân ngưỡng)", "",
              "| Kênh | Nền ↔ Lá khỏe | Lá khỏe ↔ Vết bệnh |", "|---|---|---|"]
    for ch, v in overlap.items():
        lines.append(f"| {ch} | {v['overlap_bg_leaf']:.3f} | {v['overlap_healthy_lesion']:.3f} |")

    if speed_rows:
        lines += ["", "## 5. Tốc độ (ảnh 512×512, trung vị 5 lần chạy)", "",
                  "| Phép toán | Tự code (ms) | OpenCV (ms) | Chậm hơn (lần) | Sai khác pixel lớn nhất |",
                  "|---|---|---|---|---|"]
        for r in speed_rows:
            lines.append(f"| {r['operation']} | {r['scratch_ms']} | {r['opencv_ms']} | {r['ratio']}× | "
                         f"{r['max_abs_diff']} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Phân tích Histogram & Otsu trên data/raw")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = read_metadata()
    seg_rows, cache = run_leaf_segmentation(meta)
    lesion_rows = run_lesion_otsu(cache)
    speed_rows = run_speed_benchmark(cache)

    write_csv(seg_rows, args.out / "otsu_threshold_benchmark.csv")
    write_csv(lesion_rows, args.out / "lesion_otsu_benchmark.csv")
    write_csv(speed_rows, args.out / "speed_benchmark.csv")

    plot_histograms_by_category(cache, args.out)
    overlap = plot_pixel_class_histograms(cache, args.out)
    plot_segmentation_grid(cache, "S (HSV)", args.out, seg_rows)
    if any(r["channel"] == "VegMap (pipeline)" for r in seg_rows):
        plot_segmentation_grid(cache, "VegMap (pipeline)", args.out, seg_rows)
    plot_variance_curve(cache, args.out)
    plot_lesion_examples(cache, args.out, lesion_rows)
    write_summary(seg_rows, lesion_rows, speed_rows, overlap, args.out)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
