"""
Exploded axonometric of the Murray Hill layers.

WHY THIS EXISTS
---------------
The same stack that tools/export_gis.py writes for QGIS2threejs, drawn here
directly. Two reasons to have both: this one runs in the analysis venv with
no QGIS install and regenerates from the numbers in one command, and it
keeps the layer order and z spacing honest by reading them from the same
`diagram:` block in config.yaml that the GeoPackage uses.

Every layer drawn here is measured data. Node colour is the layer's own
metric; nothing is invented to fill a plane.

    .venv/bin/python tools/figure_axo.py --out results/figures/figure_axo.png
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from common import CFG, PROC, RAW                                # noqa: E402

A = np.radians(30.0)      # axonometric half-angle; 30 deg is the reference's


def iso(x, y, z, zs):
    """Axonometric projection. z rises straight up the page, which is what
    makes a stack of planes read as a stack rather than a perspective."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x - y) * np.cos(A), (x + y) * np.sin(A) + np.asarray(z, float) * zs


def plane(ax, bounds, z, zs, pad=60, **kw):
    x0, y0, x1, y1 = bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    cx, cy = iso([x0, x1, x1, x0], [y0, y0, y1, y1], z, zs)
    ax.add_collection(PolyCollection([np.c_[cx, cy]], **kw))
    return cx, cy


def draw_nodes(ax, g, val, z, zs, cmap, accent, smin, smax, q=0.90):
    x, y = iso(g.geometry.x, g.geometry.y, z, zs)
    v = g[val].values
    r = (v - np.nanmin(v)) / max(np.ptp(v[~np.isnan(v)]), 1e-9)
    hi = v >= np.nanquantile(v, q)
    # Floor the ramp at 0.28: a Greys value of 0 is white, and a white dot
    # on a white plane is a missing node, not a low one.
    ax.scatter(x[~hi], y[~hi], s=smin + r[~hi] * (smax - smin),
               c=0.28 + r[~hi] * 0.67, cmap=cmap, vmin=0, vmax=1,
               linewidths=0, alpha=.9, zorder=3)
    # The top decile in the accent colour. The reference drawing uses red
    # for the thing being argued about; here that is the high end of the
    # layer's own metric, not a separate dataset.
    ax.scatter(x[hi], y[hi], s=smin + r[hi] * (smax - smin) * 1.6,
               c=accent, linewidths=0, alpha=.95, zorder=4)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/figure_axo.png")
    ap.add_argument("--z-scale", type=float, default=6.0,
                    help="drawing-only exaggeration of the config z spacing")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    D = CFG["diagram"]; Z = D["layer_z_m"]; acc = D["accent"]; zs = a.z_scale

    m = gpd.read_file(PROC / "metrics.gpkg").to_crs("EPSG:32618")
    fp = gpd.read_file(RAW / "building_footprints.geojson").to_crs("EPSG:32618")
    gis = Path("results/gis/murrayhill_layers.gpkg")
    faces = gpd.read_file(gis, layer="faces") if gis.exists() else None
    B = m.total_bounds

    fig, ax = plt.subplots(figsize=(13.5, 21))
    ax.set_facecolor("white")

    # ---- leader lines first, so every layer sits on top of them --------
    seg = []
    for r in m.itertuples():
        xa, ya = iso(r.geometry.x, r.geometry.y, Z["fabric"], zs)
        xb, yb = iso(r.geometry.x, r.geometry.y, Z["green"], zs)
        seg.append([(float(xa), float(ya)), (float(xb), float(yb))])
    ax.add_collection(LineCollection(seg, colors="#000000", linewidths=.16,
                                     alpha=.13, zorder=1))

    # ---- L0 fabric -----------------------------------------------------
    plane(ax, B, Z["fabric"], zs, facecolors="none",
          edgecolors="#b9b9b9", linewidths=.7, zorder=2)
    polys = []
    for geom in fp.geometry:
        gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in gs:
            cx, cy = iso(*np.array(p.exterior.coords).T, Z["fabric"], zs)
            polys.append(np.c_[cx, cy])
    ax.add_collection(PolyCollection(polys, facecolors="#e9e9e9",
                                     edgecolors="#9a9a9a", linewidths=.22,
                                     zorder=2.5))

    # ---- L1 block faces: the unit the statistics cluster on ------------
    plane(ax, B, Z["faces"], zs, facecolors="none",
          edgecolors="#b9b9b9", linewidths=.7, zorder=5)
    if faces is not None and len(faces):
        hi = faces.GVI >= faces.GVI.quantile(.80)
        for sel, col, lw in [(~hi, "#8a8a8a", 1.1), (hi, acc, 2.6)]:
            ls = []
            for geom in faces[sel].geometry:
                cx, cy = iso(*np.array(geom.coords).T, Z["faces"], zs)
                ls.append(np.c_[cx, cy])
            if ls:
                ax.add_collection(LineCollection(ls, colors=col,
                                                 linewidths=lw, zorder=6))

    # ---- L2 enclosure --------------------------------------------------
    plane(ax, B, Z["enclosure"], zs, facecolors="none",
          edgecolors="#b9b9b9", linewidths=.7, zorder=7)
    draw_nodes(ax, m.dropna(subset=["VEI"]), "VEI",
               Z["enclosure"], zs, "Greys", acc, 3, 26)

    # ---- L3 greenery ---------------------------------------------------
    plane(ax, B, Z["green"], zs, facecolors="none",
          edgecolors="#b9b9b9", linewidths=.7, zorder=8)
    draw_nodes(ax, m.dropna(subset=["GVI"]), "GVI",
               Z["green"], zs, "Greys", acc, 3, 30)

    # ---- labels --------------------------------------------------------
    lab = [(Z["green"], "GREENERY", f"GVI, {m.GVI.notna().sum()} nodes"),
           (Z["enclosure"], "ENCLOSURE",
            f"VEI, {m.VEI.notna().sum()} nodes — H:W measured on {m.HW_ratio.notna().sum()}"),
           (Z["faces"], "BLOCK FACES",
            f"{0 if faces is None else len(faces)} faces — the unit inference clusters on"),
           (Z["fabric"], "BUILT FABRIC", f"{len(fp)} footprints, DOB height_roof")]
    xl = iso(B[0], B[3], 0, zs)[0] - 190
    for zz, t, sub in lab:
        yy = iso(B[0], B[3], zz, zs)[1]
        ax.text(xl, yy, t, ha="right", va="center", fontsize=11,
                weight="bold", color="#1a1a1a", family="DejaVu Sans")
        ax.text(xl, yy - 42, sub, ha="right", va="center", fontsize=7.6,
                color="#6a6a6a", family="DejaVu Sans")
    ax.text(xl, iso(B[0], B[3], Z["green"], zs)[1] + 210,
            "MURRAY HILL", ha="right", va="center", fontsize=19,
            weight="bold", color="#111111")
    ax.text(xl, iso(B[0], B[3], Z["green"], zs)[1] + 150,
            "streetscape layers, 20 m sampling", ha="right", va="center",
            fontsize=9, color="#6a6a6a")
    ax.text(xl, iso(B[0], B[3], Z["fabric"], zs)[1] - 300,
            f"red = top decile within each layer   ·   axonometric 30°   "
            f"·   z spacing ×{zs:g} for legibility",
            ha="right", va="center", fontsize=7.4, color="#8a8a8a")

    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
