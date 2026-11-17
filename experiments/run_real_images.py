"""Chạy pipeline chẩn đoán trên toàn bộ ảnh lá THẬT trong `data/raw/real/` (đánh giá định tính).

Ảnh thật không có mặt nạ Ground Truth, nên không tính IoU/Dice. Thay vào đó, so sánh cấp độ bệnh
pipeline dự đoán với KHOẢNG cấp độ ước lượng bằng mắt ghi trong `metadata_real.csv`
(cột `est_level_min` .. `est_level_max`).

Đầu ra (mặc định `experiments/results/real/`):
- `real_results.csv`        : kết quả từng ảnh (R%, cấp dự đoán, khớp/không khớp, T*, η, tỷ lệ lá/khung...).
- `real_results_grid.png`   : lưới ảnh — ảnh gốc | vùng lá (xanh) + vết bệnh (đỏ).
- `summary.md`              : bảng tổng hợp theo nhóm ảnh.

Thêm ảnh mới: chép ảnh vào `data/raw/real/` và thêm 1 dòng vào `metadata_real.csv`
(ảnh không có trong metadata vẫn được chạy, chỉ không có cột đối chiếu).

Chạy từ thư mục gốc repository:
    python -m experiments.run_real_images
"""

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import run_full_diagnosis_pipeline

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "raw" / "real"
OUT_DEFAULT = ROOT / "experiments" / "results" / "real"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_VI = {"healthy": "Lá khỏe (tự chụp)", "diseased": "Lá bệnh", "hard": "Ca khó",
            "limitation": "Giới hạn phương pháp", "": "Chưa khai báo"}


def read_metadata(real_dir: Path) -> Dict[str, Dict]:
    path = real_dir / "metadata_real.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {row["filename"]: row for row in csv.DictReader(fh)}


def _int_or_none(v: Optional[str]) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def run(real_dir: Path, out_dir: Path) -> List[Dict]:
    meta = read_metadata(real_dir)
    images = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"Không có ảnh nào trong {real_dir}")
    rows, vis = [], []
    for path in images:
        m = meta.get(path.name, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_full_diagnosis_pipeline(str(path), run_benchmark_comparison=False)
        d = r["diagnosis"]
        leaf = r["leaf_mask"] > 0
        lesion = r["morphed_disease_mask"] > 0
        lo, hi = _int_or_none(m.get("est_level_min")), _int_or_none(m.get("est_level_max"))
        pred = int(d["severity_level"])
        if lo is None or hi is None:
            match = "n/a"
        else:
            match = "khớp" if lo <= pred <= hi else ("thấp hơn" if pred < lo else "cao hơn")
        otsu = r.get("otsu_result") or {}
        rows.append({
            "filename": path.name,
            "group": m.get("group", ""),
            "background": m.get("background", ""),
            "fills_frame": m.get("fills_frame", ""),
            "est_level": "" if lo is None else (f"{lo}" if lo == hi else f"{lo}-{hi}"),
            "est_ratio_pct": "" if not m.get("est_ratio_min_pct") else
                             f"{m['est_ratio_min_pct']}-{m['est_ratio_max_pct']}",
            "pred_level": pred,
            "pred_ratio_pct": round(float(d["disease_ratio_percent"]), 2),
            "match": match,
            "num_lesions": int(d.get("num_lesions", 0)),
            "leaf_frac_of_frame": round(float(leaf.mean()), 3),
            "otsu_T": otsu.get("threshold", ""),
            "otsu_separability": round(float(otsu["separability"]), 3) if "separability" in otsu else "",
            "fallback_modules": ";".join(r.get("fallback_modules", [])),
            "source": m.get("source", ""),
        })
        rgb = r["original_rgb"]
        over = rgb.astype(np.float64) * 0.35
        over[leaf] = rgb[leaf] * 0.55 + np.array([0, 150, 0]) * 0.45
        over[lesion] = [255, 0, 0]
        vis.append((path.name, rgb, over.astype(np.uint8), rows[-1]))
        print(f"{path.name:48s} ước lượng cấp {rows[-1]['est_level'] or '?':>4s} | dự đoán cấp {pred} "
              f"R={rows[-1]['pred_ratio_pct']:6.2f}% | {match}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "real_results.csv")
    plot_grid(vis, out_dir / "real_results_grid.png")
    write_summary(rows, out_dir / "summary.md")
    return rows


def write_csv(rows: List[Dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_grid(vis, path: Path):
    ncol = 4  # 2 cặp (gốc | kết quả) mỗi hàng
    pairs = len(vis)
    nrow = int(np.ceil(pairs / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    axes = np.atleast_2d(axes)
    for i, (name, rgb, over, row) in enumerate(vis):
        r, c = divmod(i, 2)
        a0, a1 = axes[r, 2 * c], axes[r, 2 * c + 1]
        a0.imshow(rgb)
        a0.set_title(f"{Path(name).stem}\nước lượng: cấp {row['est_level'] or '?'} ({row['est_ratio_pct'] or '?'}%)",
                     fontsize=8)
        a1.imshow(over)
        color = {"khớp": "#1a7f37", "thấp hơn": "#b35900", "cao hơn": "#b30000"}.get(row["match"], "black")
        a1.set_title(f"dự đoán: cấp {row['pred_level']}  R = {row['pred_ratio_pct']}%\n[{row['match']}]",
                     fontsize=8, color=color)
        a0.axis("off")
        a1.axis("off")
    for ax in axes.ravel()[2 * pairs:]:
        ax.axis("off")
    fig.suptitle("Pipeline trên ảnh lá thật — xanh lá: vùng lá, đỏ: vết bệnh", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def write_summary(rows: List[Dict], path: Path):
    lines = ["# Kết quả pipeline trên ảnh lá thật (`data/raw/real/`)", "",
             "Tự sinh bởi `python -m experiments.run_real_images`. Ảnh thật không có Ground Truth: "
             "cột *ước lượng* là đánh giá bằng mắt, chỉ mang tính tham khảo.", "",
             "| Nhóm | Số ảnh | Khớp | Thấp hơn | Cao hơn |", "|---|---|---|---|---|"]
    for g in ("healthy", "diseased", "hard", "limitation", ""):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        cnt = lambda k: sum(r["match"] == k for r in rs)
        lines.append(f"| {GROUP_VI.get(g, g)} | {len(rs)} | {cnt('khớp')} | {cnt('thấp hơn')} | {cnt('cao hơn')} |")
    lines += ["", "| Ảnh | Nhóm | Ước lượng (cấp / R%) | Dự đoán (cấp / R%) | Kết quả | Lá/khung |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['filename']} | {r['group']} | {r['est_level'] or '?'} / {r['est_ratio_pct'] or '?'} | "
                     f"{r['pred_level']} / {r['pred_ratio_pct']} | {r['match']} | {r['leaf_frac_of_frame']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Chạy pipeline trên ảnh lá thật")
    ap.add_argument("--input", type=Path, default=REAL_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    run(args.input, args.out)
    print(f"\nĐã ghi kết quả vào {args.out}")


if __name__ == "__main__":
    main()
