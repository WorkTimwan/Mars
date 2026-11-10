#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
terrain_compare.py — порівняння рельєфу Марс / Земля / Місяць за CSV-даними.

Встановлення залежностей:
    pip install numpy pandas scipy matplotlib rasterio

Команди:
  1) GeoTIFF (Mars Trek / topoView / Quickmap) -> CSV:
       python terrain_compare.py csv elevation.tif mars.csv --body mars

  2) Пошук ділянок, подібних до запиту (працює з 1..N csv):
       python terrain_compare.py sim mars.csv earth.csv moon.csv \
            --qlat -22.5 --qlon -68.5 --qrad 25 --pct 10
     (запит: lat/lon центру еталонної ділянки, qrad — радіус у км,
      pct — скільки % найподібніших підсвітити)

  3) Демо на вашому зображенні slope-75s-large-1.jpg (без GeoTIFF):
       python terrain_compare.py demo slope-75s-large-1.jpg demo_out
"""

import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 1) GeoTIFF -> CSV ----------
def cmd_csv(args):
    import rasterio
    with rasterio.open(args.src) as ds:
        band = ds.read(1).astype(float)
        msk = ds.read_masks(1)
        band[msk == 0] = np.nan
        rows, cols = np.indices(band.shape)
        xs, ys = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel())
    df = pd.DataFrame({"lat": np.asarray(ys), "lon": np.asarray(xs),
                       "value": band.ravel(), "body": args.body}).dropna()
    df.to_csv(args.dst, index=False)
    print(f"OK: {len(df)} точок -> {args.dst}")


# ---------- ознаки рельєфу ----------
def compute_features(grid, win):
    m = np.nanmean(grid)
    v = np.where(np.isnan(grid), m, grid)
    mean_f = ndimage.uniform_filter(v, win)                 # середнє значення
    rough  = ndimage.generic_filter(v, np.std, size=win)      # шорсткість
    grad   = np.hypot(*np.gradient(v))                        # градієнт
    return np.dstack([mean_f, rough, grad])


def to_grid(g):
    lats = np.sort(g.lat.unique())
    lons = np.sort(g.lon.unique())
    v = g.pivot_table(index="lat", columns="lon", values="value").values
    return lats, lons, v


# ---------- 2) пошук подібних ділянок ----------
def cmd_sim(args):
    dfs = [pd.read_csv(f) for f in args.csvs]
    alldf = pd.concat(dfs, ignore_index=True)
    grids = {b: to_grid(g) for b, g in alldf.groupby("body")}

    # спільне z-нормування в просторі ознак (порівняння між тілами)
    Z, F = {}, []
    win = args.win
    for b, (lats, lons, v) in grids.items():
        z = compute_features(v, win)
        Z[b] = z
        F.append(z.reshape(-1, 3))
    F = np.concatenate(F)
    mu, sd = np.nanmean(F, 0), np.nanstd(F, 0)
    Z = {b: (z - mu) / sd for b, z in Z.items()}

    # вектор запиту
    body_q = alldf.iloc[((alldf.lat - args.qlat).abs() +
                         (alldf.lon - args.qlon).abs()).idxmin()].body
    lats, lons, v = grids[body_q]
    iy = int(np.argmin(np.abs(lats - args.qlat)))
    ix = int(np.argmin(np.abs(lons - args.qlon)))
    dlat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.01
    rpx = max(1, int(args.qrad / 111.0 / dlat))
    Y, X = np.ogrid[:v.shape[0], :v.shape[1]]
    pm = (Y - iy) ** 2 + (X - ix) ** 2 <= rpx ** 2
    qvec = np.nanmean(Z[body_q][pm], axis=0)
    print(f"Запит: body={body_q}, lat={args.qlat}, lon={args.qlon}, "
          f"вікно {pm.sum()} пікселів")

    for b, z in Z.items():
        lats, lons, v = grids[b]
        D = np.linalg.norm(z - qvec, axis=2)
        thr = np.nanpercentile(D, args.pct)
        sim = D <= thr
        out = pd.DataFrame({"lat": np.repeat(lats, v.shape[1]),
                            "lon": np.tile(lons, v.shape[0]),
                            "dist": D.ravel(),
                            "similar": sim.ravel()})
        out.to_csv(f"similar_{b}.csv", index=False)
        plt.figure(figsize=(10, 5))
        plt.imshow(v, cmap="gray")
        plt.contourf(sim, levels=[0.5, 1.5], colors=["none", "red"], alpha=0.55)
        plt.title(f"{b}: ділянки, подібні до запиту ({args.pct}% найближчих)")
        plt.colorbar(label="value")
        plt.savefig(f"map_{b}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  {b}: подібних пікселів {int(sim.sum())} -> similar_{b}.csv, map_{b}.png")


# ---------- 3) демо: декодування кольорової шкали + подібність ----------
def cmd_demo(args):
    from PIL import Image
    a = np.array(Image.open(args.img).convert("RGB")).astype(int)
    h, w, _ = a.shape
    is_color = (a.max(axis=2) - a.min(axis=2)) > 40
    # рядок шкали = перший рядок із довгим кольоровим фрагментом
    counts = is_color.sum(axis=1)
    yc = int(np.argmax(counts[:h // 3] > 0.5 * counts.max()))
    row = is_color[yc]
    xs = np.where(row)[0]
    x0, x1 = xs.min(), xs.max()
    print(f"Шкала знайдена: рядок {yc}, x {x0}..{x1}")

    lut_cols = a[yc, x0:x1 + 1].astype(float)
    tree = cKDTree(lut_cols)
    flat = a.reshape(-1, 3).astype(float)
    dist, idx = tree.query(flat, k=1)
    val = idx.reshape(h, w).astype(float)
    dist = dist.reshape(h, w)
    vmax = float(args.vmax)
    value = val / max(1, (x1 - x0)) * vmax
    valid = (dist < 30) & (np.indices((h, w))[0] > yc + 5)
    print(f"Валідних пікселів: {valid.sum()}; діапазон "
          f"{value[valid].min():.1f}..{value[valid].max():.1f}")

    z = compute_features(value, args.win)
    mu, sd = z[valid].mean(0), z[valid].std(0)
    Z = (z - mu) / sd

    qy, qx = args.qy, args.qx
    rr = args.qr
    Y, X = np.ogrid[:h, :w]
    qm = (Y - qy) ** 2 + (X - qx) ** 2 <= rr ** 2
    qvec = Z[qm].mean(0)
    D = np.linalg.norm(Z - qvec, axis=2)
    thr = np.percentile(D[valid], args.pct)
    sim = valid & (D <= thr)
    print(f"Подібних: {sim.sum()} ({100*sim.sum()/valid.sum():.1f}%)")

    base = (plt.get_cmap("gray")(plt.Normalize(0, vmax)(value))[:, :, :3] * 255).astype(np.uint8)
    over = base.copy()
    over[sim] = [255, 0, 255]
    over[qm] = [0, 255, 255]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(base); ax[0].set_title(f"Розкодоване значення (0..{vmax})")
    ax[0].add_patch(plt.Circle((qx, qy), rr, color="cyan", fill=False, lw=2))
    ax[1].imshow(over); ax[1].set_title("Подібні ділянки (магента), запит (блакитний)")
    for a_ in ax: a_.axis("off")
    os.makedirs(args.out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "demo_result.png"), dpi=110, bbox_inches="tight")
    # CSV результату
    yy, xx = np.indices((h, w))

    pd.DataFrame({
        "x": xx[sim],
        "y": yy[sim],
        "value": value[sim],
        "dist": D[sim]
    }).to_csv(
        os.path.join(args.out, "demo_similar.csv"), index=False)
    print(f"Збережено: {args.out}/demo_result.png та demo_similar.csv")


def main():
    p = argparse.ArgumentParser(description="Порівняння рельєфу планет за CSV")
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("csv", help="GeoTIFF -> CSV")
    c1.add_argument("src"); c1.add_argument("dst"); c1.add_argument("--body", default="mars")
    c1.set_defaults(func=cmd_csv)

    c2 = sub.add_parser("sim", help="пошук подібних ділянок за CSV")
    c2.add_argument("csvs", nargs="+")
    c2.add_argument("--qlat", type=float, required=True)
    c2.add_argument("--qlon", type=float, required=True)
    c2.add_argument("--qrad", type=float, default=25, help="радіус запиту, км")
    c2.add_argument("--pct", type=float, default=10, help="% найподібніших")
    c2.add_argument("--win", type=int, default=9, help="вікно ознак, пікселів")
    c2.set_defaults(func=cmd_sim)

    c3 = sub.add_parser("demo", help="демо на картинці зі шкалою (як slope-75s)")
    c3.add_argument("img")
    c3.add_argument("out", nargs="?", default="demo_out")
    c3.add_argument("--vmax", type=float, default=32, help="макс. значення шкали")
    c3.add_argument("--qy", type=int, default=1950, help="y запиту")
    c3.add_argument("--qx", type=int, default=770, help="x запиту")
    c3.add_argument("--qr", type=int, default=40, help="радіус запиту, пікс")
    c3.add_argument("--pct", type=float, default=12)
    c3.add_argument("--win", type=int, default=25)
    c3.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()