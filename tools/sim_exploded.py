"""Exploded axonometric: the SIM's three dimensions as strata over the plan.

The Cobb-Douglas is a product of three terms, M = I^a Y^b D^c, and the natural
drawing of a product is a stack: each dimension gets its own plane over the
same street network, and the composite sits above them. Vertical threads tie a
node's position through every plane, so a street that is dark in one stratum
and bright in another can be read down the column.

WHY A HAND-ROLLED PROJECTION rather than mplot3d. The axes-3d renderer sorts
whole artists rather than segments, so a street on one plane will pop in front
of a plane above it, and there is no way to make thin lines crisp. The
projection here is a plain isometric shear on 2-D coordinates: the layers are
drawn back to front, so occlusion is exactly the draw order.

    x' = (x - y) cos(30)          y' = (x + y) sin(30) + layer_gap * k

EACH STRATUM IS ONE CONTINUOUS SURFACE. The nodes are samples of a field that
exists everywhere between them, so the honest drawing is a warped sheet, not a
row of uprights and not a ribbon per street: values are interpolated onto a
grid, smoothed, and the sheet is lifted by the value at each cell. The top of
the sheet is the dimension's relief across the study area.

INTERPOLATION IS BOUNDED BY DISTANCE, not by the hull. A cubic fit across the
whole convex hull would invent values in the middle of blocks where no frontage
was ever photographed, and those inventions would be the largest, smoothest
features in the picture. Cells further than `--reach` from any node are
dropped, so the sheet spreads from the streets and stops.

PAINTER'S ALGORITHM, BY CELL. In this projection up-screen is further away, so
each sheet's cells are drawn in order of decreasing screen height. Sorting
whole layers is not enough once a sheet has relief: a tall cell at the front of
a plane must cover a low cell behind it in the same plane.

No accent colour. An overlaid mark for the top decile sat outside every ramp
and pulled the eye off the surfaces it was meant to annotate; the ramps already
put the tail at their bright end.

Both study areas, from vlm_calculations.csv and nodes.gpkg alone. Murray Hill
has building footprints and London does not, so the fabric plane is drawn from
the street network in both -- one figure that means the same thing twice.

    .venv/Scripts/python tools/sim_exploded.py
    SIM_CONFIG=config_london.yaml .venv/Scripts/python tools/sim_exploded.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.collections import LineCollection, PolyCollection

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, PROJ_CRS, banner

import cmcrameri.cm as cmc
import cmocean

BG, INK, MUT = "#ffffff", "#1a1a1a", "#8a8a8a"
RED = "#c0392b"
GAP_M = 60.0          # a break longer than this is a gap, not a street

# bottom to top: the fabric, then the three terms, then what they make
LAYERS = [
    ("fabric", "the street network",  None,           None),
    ("D_raw",  "place dependence",    cmocean.cm.ice, "D"),
    ("Y",      "place identity",      cmc.lajolla,    "Y"),
    ("I_raw",  "place imageability",  "viridis",      "I"),
    ("M",      "the composite",       "magma",        "M"),
]


def iso(x, y, k, gap):
    """Isometric shear, with layer k lifted by `gap` in projected units."""
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    return (x - y) * c, (x + y) * s + k * gap


def chains(g):
    """Consecutive node pairs along each street, as index pairs."""
    out = []
    key = "chain" if g.chain.notna().any() else "osm_name"
    for _, d in g.groupby(key):
        d = d.sort_values("chain_pos_m" if d.chain_pos_m.notna().all() else "_i")
        idx = d["_i"].to_numpy()
        xy = np.c_[d._x.to_numpy(), d._y.to_numpy()]
        seg = np.hypot(*(xy[1:] - xy[:-1]).T)
        for a, b, s in zip(idx[:-1], idx[1:], seg):
            if s <= GAP_M:
                out.append((a, b))
    return np.array(out) if out else np.zeros((0, 2), int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", type=Path, default=None)
    ap.add_argument("--nodes", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gap", type=float, default=0.62,
                    help="layer separation, as a fraction of the frame's span")
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--context-km", type=float, default=1.6,
                    help="radius of the wider city plan drawn behind the "
                         "stack; 0 to omit")
    ap.add_argument("--bars", action="store_true", default=True,
                    help="extrude each node's value as a bar within its plane")
    ap.add_argument("--relief", type=float, default=0.20,
                    help="height of the surface at full value, as a fraction "
                         "of the frame span")
    ap.add_argument("--grid", type=int, default=190,
                    help="interpolation grid, cells across the frame")
    ap.add_argument("--reach", type=float, default=55.0,
                    help="metres from a node beyond which the surface stops")
    ap.add_argument("--smooth", type=float, default=1.6,
                    help="gaussian smoothing of the field, in grid cells")
    args = ap.parse_args()
    name = CFG.get("study_area_name", "study area")
    banner(f"exploded axonometric: {name}")

    calc = args.calc or RES / "tables" / "vlm_calculations.csv"
    nodes = args.nodes or PROC / "nodes.gpkg"
    c = pd.read_csv(calc)
    mcol = "M_noA" if "M_noA" in c.columns else "M"
    keep = [v[0] for v in LAYERS if v[0] != "fabric"]
    per = c.groupby("node_id")[[k if k != "M" else mcol
                                for k in keep]].mean().reset_index()
    per = per.rename(columns={mcol: "M"})

    g = gpd.read_file(nodes).to_crs(PROJ_CRS)
    g = g.merge(per, on="node_id", how="inner")
    g["_x"], g["_y"] = g.geometry.x.values, g.geometry.y.values
    # keep the origin: the context plan and the boundary must be shifted by the
    # same amount, and recomputing the mean after centring gives zero
    OX, OY = float(g._x.mean()), float(g._y.mean())
    g["_x"] -= OX; g["_y"] -= OY
    g = g.reset_index(drop=True)
    g["_i"] = np.arange(len(g))
    print(f"{len(g)} nodes with a score, M column {mcol}")

    pairs = chains(g)
    print(f"{len(pairs)} street segments")
    span = max(np.ptp(g._x.values), np.ptp(g._y.values))
    gap = args.gap * span

    fig, ax = plt.subplots(figsize=(9.0, 16.4), facecolor=BG)
    ax.set_facecolor(BG); ax.set_axis_off()

    # The wider plan, on the ground plane and behind everything. It is what
    # gives the stack a place to stand: without it the study area floats and
    # the reader cannot tell whether it is a whole city or six streets of one.
    # Cached, because the fetch is slow and the answer never changes.
    if args.context_km > 0:
        cache = PROC / f"context_{int(args.context_km*1000)}m.gpkg"
        if cache.exists():
            ctx = gpd.read_file(cache).to_crs(PROJ_CRS)
        else:
            import osmnx as ox
            gg = gpd.read_file(nodes).to_crs(4326)
            cx, cy = gg.geometry.x.mean(), gg.geometry.y.mean()
            print(f"fetching {args.context_km:g} km of context network ...")
            G = ox.graph_from_point((cy, cx), dist=args.context_km * 1000,
                                    network_type="all", simplify=True)
            ctx = ox.graph_to_gdfs(G, nodes=False)[["geometry"]]
            ctx.to_crs(4326).to_file(cache, driver="GPKG")
            ctx = ctx.to_crs(PROJ_CRS)
        segs = []
        for geom in ctx.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for q in parts:
                a = np.asarray(q.coords, float)
                X, Y = iso(a[:, 0] - OX, a[:, 1] - OY, 0, gap)
                segs.extend(np.stack([np.c_[X, Y][:-1], np.c_[X, Y][1:]], axis=1))
        if segs:
            ax.add_collection(LineCollection(np.array(segs), colors="#ededed",
                                             lw=.45, zorder=-5))
            print(f"context network: {len(segs)} segments")

    # The study boundary, on the ground. The context plan says where; this says
    # how much of it was measured.
    hull = gpd.GeoSeries(g.geometry.values, crs=g.crs).union_all().convex_hull
    hb = np.asarray(hull.exterior.coords, float)
    HX, HY = iso(hb[:, 0] - OX, hb[:, 1] - OY, 0, gap)
    ax.plot(HX, HY, color="#b9b9b9", lw=1.0, ls=(0, (5, 3)), zorder=-3)
    ax.fill(HX, HY, color="#f4f4f4", zorder=-4)

    # threads first, behind everything: one per node, tying the strata together
    thin = g.sample(min(len(g), 260), random_state=3)
    for _, r in thin.iterrows():
        xs, ys = [], []
        for k in range(len(LAYERS)):
            X, Y = iso(r._x, r._y, k, gap)
            xs.append(X); ys.append(Y)
        ax.plot(xs, ys, color="#d7d7d7", lw=.35, zorder=0, solid_capstyle="butt")

    # the field, once: grid, interpolate, smooth, mask by distance
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    x0, x1 = g._x.min(), g._x.max()
    y0, y1 = g._y.min(), g._y.max()
    pad = span * 0.03
    gx = np.linspace(x0 - pad, x1 + pad, args.grid)
    gy = np.linspace(y0 - pad, y1 + pad, args.grid)
    GX, GY = np.meshgrid(gx, gy)
    cell = (gx[1] - gx[0])
    occupied = np.zeros_like(GX, bool)
    ix = np.clip(((g._x - gx[0]) / cell).astype(int), 0, args.grid - 1)
    iy = np.clip(((g._y - gy[0]) / cell).astype(int), 0, args.grid - 1)
    occupied[iy, ix] = True
    near = distance_transform_edt(~occupied) * cell <= args.reach
    print(f"surface grid {args.grid}x{args.grid}, "
          f"{near.mean()*100:.0f}% of cells within {args.reach:g} m of a node")

    pts = np.c_[g._x.to_numpy(), g._y.to_numpy()]
    for k, (col, label, cmap, short) in enumerate(LAYERS):
        if col == "fabric":
            X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
            P = np.c_[X, Y]
            segs = P[pairs] if len(pairs) else np.zeros((0, 2, 2))
            ax.add_collection(LineCollection(segs, colors="#c4c4c4", lw=.9,
                                             zorder=k * 10 + 1))
            lx, ly = X.min() - span * 0.20, Y[np.argmin(X)]
            ax.text(lx, ly, label, color=MUT, fontsize=8.2,
                    ha="right", va="center")
            continue

        v = g[col].to_numpy()
        lo, hi = np.nanpercentile(v, [3, 97])
        F = griddata(pts, v, (GX, GY), method="cubic")
        Fl = griddata(pts, v, (GX, GY), method="linear")
        F = np.where(np.isfinite(F), F, Fl)
        F = np.where(np.isfinite(F), F,
                     griddata(pts, v, (GX, GY), method="nearest"))
        F = gaussian_filter(F, args.smooth)
        F = np.where(near, F, np.nan)

        t = np.clip((F - lo) / max(hi - lo, 1e-9), 0, 1) * args.relief * span
        SX, SY = iso(GX, GY, k, gap)
        SY = SY + t

        # quads over the grid, drawn far to near within this stratum
        a = (slice(0, -1), slice(0, -1)); b = (slice(0, -1), slice(1, None))
        c_ = (slice(1, None), slice(1, None)); d_ = (slice(1, None), slice(0, -1))
        val = np.nanmean(np.stack([F[a], F[b], F[c_], F[d_]]), axis=0)
        ok = np.isfinite(val)
        quads = np.stack([np.stack([SX[q], SY[q]], -1) for q in (a, b, c_, d_)], -2)
        quads = quads[ok]; val = val[ok]
        depth = quads[:, :, 1].mean(axis=1)
        o = np.argsort(-depth)
        ax.add_collection(PolyCollection(
            quads[o], cmap=cmap, array=val[o], norm=plt.Normalize(lo, hi),
            linewidths=0.28, edgecolors="none", zorder=k * 10 + 1))

        X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
        lx, ly = X.min() - span * 0.20, Y[np.argmin(X)]
        ax.text(lx, ly, label, color=INK, fontsize=8.2, ha="right", va="center")
        ax.text(lx, ly - span * 0.05, f"median {g[col].median():.3f}",
                color=MUT, fontsize=6.4, ha="right", va="center")

    ax.autoscale_view()
    ax.set_aspect("equal")
    fig.text(.06, .965, name, color=INK, fontsize=17)
    fig.text(.06, .945,
             f"M = I^a · Y^b · D^c, drawn as strata.  {len(g)} nodes.  "
             f"Each stratum is one interpolated surface; height and colour both carry the value.",
             color=MUT, fontsize=8.6)
    out = args.out or RES / "figures" / f"sim_exploded_{CFG.get('study_area_slug','area')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
