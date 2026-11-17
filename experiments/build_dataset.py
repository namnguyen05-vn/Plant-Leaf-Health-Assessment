"""Sinh bộ dữ liệu chuẩn có Ground Truth cho dự án (Vai trò 2 — Xây dựng Dữ liệu).

Bộ ảnh mô phỏng (procedural) lá cây chụp trên nhiều loại nền, có điều khiển chính xác:
- Mức độ bệnh theo đúng thang 5 cấp của `src/benchmark.diagnose_leaf_severity`
  (Cấp 0 <1%, Cấp 1 1–10%, Cấp 2 10–25%, Cấp 3 25–50%, Cấp 4 >=50% diện tích phiến lá).
- Các điều kiện khó (failure cases): bóng đổ, chói sáng, nhiễu hạt, gân lá vàng, thiếu sáng.
- Mỗi ảnh đi kèm mặt nạ Ground Truth: phiến lá (leaf) và vết bệnh (lesion = lõi hoại tử + quầng vàng)
  -> dùng trực tiếp để tính IoU / Dice / sai số tỷ lệ bệnh R% trong `src/benchmark.py`.

Vì sao dùng ảnh mô phỏng: ảnh thật (vd. PlantVillage) không có mặt nạ vết bệnh chuẩn, nên không
đo định lượng được. Ảnh mô phỏng cho phép đo chính xác; ảnh thật nên được BỔ SUNG vào `data/raw/real/`
để đánh giá định tính (xem README trong data/raw).

Chỉ dùng NumPy + Pillow (lưu file). Kết quả tái lập 100% nhờ seed cố định.

Chạy từ thư mục gốc repository:
    python -m experiments.build_dataset              # sinh vào data/raw/
    python -m experiments.build_dataset --out tmp_ds  # thư mục khác
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.data_io import save_matrix_to_csv
from src.preprocess import rgb_to_gray_scratch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw"


# =============================================================================
# CẤU HÌNH BỘ DỮ LIỆU
# =============================================================================

@dataclass
class Spec:
    name: str
    category: str               # healthy / mild / moderate / severe / critical / challenge
    target_ratio: Tuple[float, float]  # khoảng tỷ lệ bệnh mục tiêu (%) trên phiến lá
    background: str
    condition: str = "normal"   # normal / shadow / glare / noise / yellow_vein / lowlight
    seed: int = 0
    portrait: bool = False
    edge_blight: bool = False   # thêm mảng cháy lá từ mép (bệnh nặng)
    extra: Dict = field(default_factory=dict)


def build_specs() -> List[Spec]:
    specs: List[Spec] = []
    bgs = ["paper", "soil", "wood", "dark", "gray"]

    def add(prefix, n, category, ratio, start_seed, **kw):
        for i in range(n):
            specs.append(Spec(
                name=f"{prefix}_{i + 1:02d}", category=category, target_ratio=ratio,
                background=bgs[(start_seed + i) % len(bgs)], seed=start_seed * 100 + i,
                portrait=(i % 3 == 2), **kw,
            ))

    add("healthy", 5, "healthy", (0.0, 0.0), 1)
    add("mild", 6, "mild", (2.0, 8.0), 2)
    add("moderate", 5, "moderate", (12.0, 22.0), 3)
    add("severe", 4, "severe", (28.0, 45.0), 4)
    for i in range(2):
        specs.append(Spec(f"critical_{i + 1:02d}", "critical", (55.0, 70.0), bgs[i + 1],
                          seed=500 + i, edge_blight=True))
    specs[-6].edge_blight = True  # một ca severe có cháy mép lá

    challenge = [
        ("shadow_01", (3.0, 8.0), "paper", "shadow"),
        ("shadow_02", (12.0, 20.0), "soil", "shadow"),
        ("glare_01", (0.0, 0.0), "dark", "glare"),
        ("glare_02", (4.0, 9.0), "gray", "glare"),
        ("noise_01", (3.0, 8.0), "wood", "noise"),
        ("noise_02", (14.0, 22.0), "paper", "noise"),
        ("yellowvein_01", (0.0, 0.0), "gray", "yellow_vein"),
        ("lowlight_01", (5.0, 9.0), "soil", "lowlight"),
    ]
    for i, (name, ratio, bg, cond) in enumerate(challenge):
        specs.append(Spec(name, "challenge", ratio, bg, condition=cond, seed=700 + i, portrait=(i % 2 == 1)))
    return specs


# =============================================================================
# HÀM VẼ
# =============================================================================

def smooth_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    """Nhiễu tần số thấp trong [-1, 1] (lưới ngẫu nhiên phóng to song tuyến)."""
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    grid = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    big = Image.fromarray(grid).resize((w, h), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(big, dtype=np.float64), -1.5, 1.5)


def make_background(rng, kind: str, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n1 = smooth_noise(rng, h, w, 48)[..., None]
    n2 = smooth_noise(rng, h, w, 8)[..., None]
    grad = (1.0 + 0.12 * ((xx / w) - 0.5) * rng.choice([-1, 1]) + 0.08 * ((yy / h) - 0.5))[..., None]
    if kind == "paper":
        base = np.array([232, 228, 218]) + rng.uniform(-8, 8, 3)
        img = base + 6 * n1 + 3 * n2
    elif kind == "soil":
        base = np.array([112, 82, 58]) + rng.uniform(-10, 10, 3)
        grain = rng.normal(0, 14, (h, w, 1))
        img = base + 18 * n1 + 12 * n2 + grain
    elif kind == "wood":
        base = np.array([150, 108, 70]) + rng.uniform(-10, 10, 3)
        stripes = np.sin((yy + 6 * smooth_noise(rng, h, w, 60)) / rng.uniform(5, 9))[..., None]
        img = base + 16 * stripes + 10 * n1
    elif kind == "dark":
        base = np.array([38, 38, 44]) + rng.uniform(-5, 5, 3)
        img = base + 5 * n1 + 3 * n2
    else:  # gray table
        base = np.array([150, 150, 148]) + rng.uniform(-10, 10, 3)
        img = base + 8 * n1 + 4 * n2
    return np.clip(img * grad, 0, 255)


def leaf_geometry(rng, h: int, w: int):
    """Trả về các trường tọa độ cục bộ (s dọc gân chính, t vuông góc) và mặt nạ phiến/cuống."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = w / 2 + rng.uniform(-0.08, 0.08) * w
    cy = h / 2 + rng.uniform(-0.08, 0.08) * h
    length = rng.uniform(0.62, 0.80) * min(h, w) * (1.25 if h != w else 1.0)
    length = min(length, 0.88 * max(h, w))
    half_len = length / 2
    width_ratio = rng.uniform(0.32, 0.48)
    half_wid = half_len * width_ratio * 1.25
    theta = rng.uniform(0, np.pi) if h == w else (rng.uniform(-0.5, 0.5) + (0 if w > h else np.pi / 2))
    c, s_ = np.cos(theta), np.sin(theta)
    s = (xx - cx) * c + (yy - cy) * s_    # trục dọc lá
    t = -(xx - cx) * s_ + (yy - cy) * c   # trục ngang lá
    u = s / half_len                      # [-1 gốc, 1 chóp]

    a, b = rng.uniform(0.45, 0.75), rng.uniform(0.9, 1.3)  # độ tròn gốc / độ nhọn chóp
    base = np.clip((1 + u) / 2, 0, 1) ** a * np.clip((1 - u) / 2, 0, 1) ** b
    peak = (a / (a + b)) ** a * (b / (a + b)) ** b
    half_width = half_wid * base / peak
    serr_n = rng.integers(14, 30)
    serr_amp = rng.uniform(0.0, 0.05)
    half_width *= 1 + serr_amp * np.abs(np.sin(serr_n * np.pi * u))
    asym = 1 + rng.uniform(-0.08, 0.08) * np.sign(t)
    blade = (np.abs(u) <= 1) & (np.abs(t) <= half_width * asym)

    pet_len = rng.uniform(0.10, 0.22) * half_len
    pet_w = max(2.0, 0.012 * min(h, w))
    petiole = (s < -half_len + 2) & (s > -half_len - pet_len) & (np.abs(t + 0.04 * (s + half_len)) <= pet_w)
    return {"s": s, "t": t, "u": u, "half_len": half_len, "half_width": half_width,
            "blade": blade, "petiole": petiole & ~blade}


def paint_leaf(rng, geo, h, w, yellow_vein: bool) -> np.ndarray:
    base = np.array([rng.uniform(45, 85), rng.uniform(115, 165), rng.uniform(30, 60)])
    n1 = smooth_noise(rng, h, w, 40)[..., None]
    n2 = smooth_noise(rng, h, w, 6)[..., None]
    color = base * (1 + 0.10 * n1 + 0.04 * n2)
    # Mép lá hơi sẫm, gần gân chính hơi sáng
    rel = np.clip(np.abs(geo["t"]) / np.maximum(geo["half_width"], 1), 0, 1)[..., None]
    color = color * (1.06 - 0.14 * rel ** 2)

    # Gân lá: gân chính + gân phụ xiên
    s, t = geo["s"], geo["t"]
    vein_w = max(1.0, 0.0045 * min(h, w)) * (1.8 if yellow_vein else 1.0)
    midrib = (np.abs(t) <= vein_w * 1.3 * (1 - 0.5 * np.clip(geo["u"], 0, 1)))
    lateral = np.zeros_like(midrib)
    n_lat = rng.integers(5, 9)
    for k in range(n_lat):
        s_k = -geo["half_len"] * 0.8 + (k + 0.5) * 1.6 * geo["half_len"] / n_lat
        lateral |= np.abs(s - s_k - 0.75 * np.abs(t)) <= vein_w * 0.7
    veins = (midrib | lateral) & geo["blade"]
    if yellow_vein:
        vein_color = np.array([205, 196, 78])
        color = np.where(veins[..., None], 0.15 * color + 0.85 * vein_color, color)
    else:
        color = np.where(veins[..., None], color * 1.18 + np.array([12, 10, 0]), color)
    return color


def stamp_lesion(rng, geo, h, w, cy, cx, r0, core, halo, color, allow_outside=False):
    """Vẽ một đốm bệnh hình dạng bất quy tắc: lõi nâu hoại tử + quầng vàng."""
    halo_k = rng.uniform(0.25, 0.55)
    R = int(np.ceil(r0 * (1 + halo_k) * 1.4)) + 2
    y0, y1 = max(0, cy - R), min(h, cy + R + 1)
    x0, x1 = max(0, cx - R), min(w, cx + R + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    rho = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    rad = r0 * (1 + 0.22 * np.sin(2 * ang + rng.uniform(0, 6.3)) + 0.12 * np.sin(3 * ang + rng.uniform(0, 6.3))
                + 0.06 * np.sin(5 * ang + rng.uniform(0, 6.3)))
    blade = geo["blade"][y0:y1, x0:x1]
    c_mask = (rho <= rad) & blade
    h_mask = (rho <= rad * (1 + halo_k)) & blade & ~c_mask

    brown = np.array([rng.uniform(95, 145), rng.uniform(55, 85), rng.uniform(18, 40)])
    yellow = np.array([rng.uniform(185, 215), rng.uniform(170, 200), rng.uniform(45, 85)])
    rel = np.clip(rho / np.maximum(rad, 1e-6), 0, 1)[..., None]
    core_col = brown * (0.62 + 0.38 * rel)  # tâm sẫm hơn
    ring = (np.sin(rho / max(1.5, r0 * 0.18)) > 0.6)[..., None]
    core_col = np.where(ring & (rel > 0.3), core_col * 0.88, core_col)
    sub = color[y0:y1, x0:x1]
    hrel = np.clip((rho / np.maximum(rad, 1e-6) - 1) / halo_k, 0, 1)[..., None]
    halo_col = (0.85 - 0.35 * hrel) * yellow + (0.15 + 0.35 * hrel) * sub
    sub[:] = np.where(c_mask[..., None], core_col, np.where(h_mask[..., None], halo_col, sub))
    core[y0:y1, x0:x1] |= c_mask
    halo[y0:y1, x0:x1] |= h_mask


def add_lesions(rng, geo, color, spec: Spec, h, w):
    blade = geo["blade"]
    leaf_area = blade.sum()
    core = np.zeros((h, w), bool)
    halo = np.zeros((h, w), bool)
    lo, hi = spec.target_ratio
    if hi <= 0:
        return core, halo
    target = rng.uniform(lo, hi)
    ys, xs = np.nonzero(blade)
    big = target >= 25
    if spec.edge_blight:
        # Mảng cháy lá lan từ mép chóp lá
        tip = np.argmax(geo["u"] * blade - 10 * ~blade)
        ty, tx = np.unravel_index(tip, (h, w))
        stamp_lesion(rng, geo, h, w, ty, tx, rng.uniform(0.25, 0.35) * geo["half_len"], core, halo, color)
    tries = 0
    while (core | halo).sum() / leaf_area * 100 < target and tries < 400:
        tries += 1
        i = rng.integers(len(ys))
        scale = np.sqrt(leaf_area)
        r0 = rng.uniform(0.04, 0.10 if big else 0.06) * scale
        if target < 10:
            r0 = rng.uniform(0.015, 0.045) * scale
        # Không vượt quá mục tiêu quá nhiều ở cấp nhẹ
        before = (core | halo).sum()
        snap_core, snap_halo, snap_color = core.copy(), halo.copy(), color.copy()
        stamp_lesion(rng, geo, h, w, ys[i], xs[i], r0, core, halo, color)
        after = (core | halo).sum() / leaf_area * 100
        if after > hi + 1.0 and before > 0:
            core, halo = snap_core, snap_halo
            color[:] = snap_color
            if r0 < 0.02 * scale:
                break
    return core, halo


def apply_condition(rng, img, geo, cond: str, h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if cond == "shadow":
        ang = rng.uniform(0, 2 * np.pi)
        d = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang) - rng.uniform(-0.1, 0.15) * min(h, w)
        soft = 1 / (1 + np.exp(-d / rng.uniform(6, 14)))
        img = img * (1 - rng.uniform(0.45, 0.6) * soft)[..., None]
    elif cond == "glare":
        for _ in range(rng.integers(1, 3)):
            ys, xs = np.nonzero(geo["blade"])
            k = rng.integers(len(ys))
            sy, sx = rng.uniform(0.05, 0.10) * h, rng.uniform(0.08, 0.16) * w
            g = np.exp(-(((yy - ys[k]) / sy) ** 2 + ((xx - xs[k]) / sx) ** 2))
            g = (g * geo["blade"])[..., None]
            img = img + (255 - img) * 0.9 * g
    elif cond == "noise":
        img = img + rng.normal(0, 18, img.shape)
        sp = rng.random((h, w))
        img[sp < 0.006] = 0
        img[sp > 0.994] = 255
    elif cond == "lowlight":
        img = img * 0.42 + rng.normal(0, 5, img.shape)
    return img


def soften(img: np.ndarray) -> np.ndarray:
    """Làm mềm biên nhẹ (mô phỏng quang học máy ảnh) bằng box 3x3 tách được."""
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    v = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
    return (v[:, :-2] + v[:, 1:-1] + v[:, 2:]) / 3


def render(spec: Spec) -> Dict:
    rng = np.random.default_rng(spec.seed)
    h, w = (512, 384) if spec.portrait else (384, 512)
    bg = make_background(rng, spec.background, h, w)
    geo = leaf_geometry(rng, h, w)
    color = paint_leaf(rng, geo, h, w, yellow_vein=(spec.condition == "yellow_vein"))
    core, halo = add_lesions(rng, geo, color, spec, h, w)

    pet_col = np.array([95, 125, 45])
    img = np.where(geo["blade"][..., None], color, bg)
    img = np.where(geo["petiole"][..., None], pet_col, img)
    # bóng mềm của lá trên nền (chỉ nền sáng)
    if spec.background in ("paper", "gray"):
        sh = np.roll(np.roll(geo["blade"] | geo["petiole"], 6, axis=0), 5, axis=1) & ~(geo["blade"] | geo["petiole"])
        img = np.where(sh[..., None], img * 0.82, img)
    img = soften(img)
    img = apply_condition(rng, img, geo, spec.condition, h, w)
    img = img + rng.normal(0, 2.5, img.shape)  # nhiễu cảm biến
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    leaf = geo["blade"]
    lesion = (core | halo) & leaf
    ratio = lesion.sum() / max(leaf.sum(), 1) * 100
    level = 0 if ratio < 1 else 1 if ratio < 10 else 2 if ratio < 25 else 3 if ratio < 50 else 4
    n_lesions = count_components(lesion)
    return {"rgb": img, "leaf": leaf, "lesion": lesion, "core": core & leaf,
            "ratio": ratio, "level": level, "n_lesions": n_lesions, "size": (h, w)}


def count_components(mask: np.ndarray) -> int:
    """Đếm thành phần liên thông 8-lân cận (union-find vector hóa đơn giản bằng lan truyền nhãn)."""
    h, w = mask.shape
    labels = np.where(mask, np.arange(h * w).reshape(h, w) + 1, 0)
    while True:
        p = np.pad(labels, 1)
        neigh = [p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        stacked = np.stack(neigh)
        stacked = np.where(stacked > 0, stacked, np.iinfo(labels.dtype).max)
        new = np.where(mask, stacked.min(axis=0), 0)
        if np.array_equal(new, labels):
            break
        labels = new
    return int(len(np.unique(labels[labels > 0])))


# =============================================================================
# MAIN
# =============================================================================

def _save_png(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(path, optimize=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args(argv)

    out = args.out
    gt_dir = out / "ground_truth"
    csv_dir = out / "csv"
    for d in (out, gt_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    csv_samples = {"healthy_01", "mild_01", "moderate_01", "severe_01", "shadow_01"}
    for spec in build_specs():
        res = render(spec)
        Image.fromarray(res["rgb"]).save(out / f"{spec.name}.jpg", quality=args.jpeg_quality, subsampling=0)
        _save_png((res["leaf"] * 255).astype(np.uint8), gt_dir / f"{spec.name}_leaf.png")
        _save_png((res["lesion"] * 255).astype(np.uint8), gt_dir / f"{spec.name}_lesion.png")
        if spec.name in csv_samples:
            # Ma trận mức xám 128 px (nhẹ, mở được bằng Excel) + mặt nạ GT tương ứng
            small = np.asarray(Image.fromarray(res["rgb"]).resize(
                (res["size"][1] // 4, res["size"][0] // 4), Image.Resampling.BILINEAR))
            save_matrix_to_csv(rgb_to_gray_scratch(small), csv_dir / f"{spec.name}_gray_q4.csv")
            leaf_small = np.asarray(Image.fromarray((res["leaf"] * 255).astype(np.uint8)).resize(
                (res["size"][1] // 4, res["size"][0] // 4), Image.Resampling.NEAREST))
            save_matrix_to_csv(leaf_small, csv_dir / f"{spec.name}_leafmask_q4.csv")
        rows.append({
            "filename": f"{spec.name}.jpg",
            "category": spec.category,
            "condition": spec.condition,
            "background": spec.background,
            "height": res["size"][0],
            "width": res["size"][1],
            "leaf_area_px": int(res["leaf"].sum()),
            "lesion_area_px": int(res["lesion"].sum()),
            "gt_disease_ratio_pct": round(float(res["ratio"]), 3),
            "gt_severity_level": res["level"],
            "gt_num_lesions": res["n_lesions"],
            "seed": spec.seed,
        })
        print(f"{spec.name:15s} {spec.category:9s} {spec.condition:11s} R={res['ratio']:6.2f}%  "
              f"cấp {res['level']}  ổ bệnh={res['n_lesions']}")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã sinh {len(rows)} ảnh vào {out}")


if __name__ == "__main__":
    main()
