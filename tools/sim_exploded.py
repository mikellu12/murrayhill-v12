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

HEIGHT IS LOCAL, COLOUR IS GLOBAL. Read a ribbon's height against the street
line directly beneath it and nowhere else. In an axonometric the screen height
of a mark is its depth plus its elevation, so a node at the far corner of the
plan sits high on screen because it is far, and no tilt fixes this: for Park
Avenue's 0.670 to out-top Tudor City Place's 0.581 across 1,100 m of plan the
ground would have to be flatter than about one degree, which is no axonometric
at all. The value differences between streets are small against the extent.

So colour carries the comparison and height carries the texture. Anyone asking
which street scores highest should read tools/sim_maps.py, which is flat and
has no depth axis to confound.

THREE WAYS TO DRAW A STRATUM, because they are not equally readable and the
right one depends on the audience.

  ribbon   (default) a wall along each street, its top edge the profile of the
           dimension. Continuous exactly where the measurement is continuous --
           along a frontage -- and discontinuous across blocks, where the
           measurement genuinely stops. Reads as streets.
  bars     one upright per node. The rawest view: every mark is a datum and
           nothing between them is drawn at all.
  surface  values interpolated onto a grid and lifted into a sheet. Smooth, and
           the least legible: it reads as terrain, and a reader has to be told
           that the hills are streets.

The surface was tried first and abandoned for readability. It is kept because
it is the only one that shows a neighbourhood rather than a line, but two of
its properties are traps. It interpolates values into blocks where no frontage
was photographed, so `--reach` stops the sheet a fixed distance from any node.
And it must use LINEAR interpolation: a cubic fit overshoots its inputs and put
the imageability peak on the wrong street entirely.

PAINTER'S ALGORITHM, BY MARK. In this projection up-screen is further away, so
marks are drawn in order of decreasing screen height. Sorting whole layers is
not enough once a plane has relief: a tall mark at the front must cover a short
one behind it within the same plane.

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


# Ground tilt, in degrees. NOT 30, and the reason matters: in any axonometric
# the screen height of a mark is its depth plus its elevation, because both map
# to the same axis -- y' = (x + y) sin(tilt) + value. At 30 degrees the ground
# contributes so much vertical that a node in the far corner of the frame sits
# higher on screen than a tall mark in the near corner, and the reader sees a
# peak that is only distance. Murray Hill's imageability reads as highest in
# the north-east on a 30-degree ground, when the north-east is in fact its
# LOWEST quadrant, 0.351 against the south-west's 0.413.
#
# Flattening the ground shrinks the depth term without touching the value term.
# At 16 degrees the ground spans about a third of the vertical it did, so
# relief carries the height and the colour and the profile agree.
TILT = 16.0


def iso(x, y, k, gap):
    """Axonometric shear, with layer k lifted by `gap` in projected units."""
    c, s = np.cos(np.radians(30)), np.sin(np.radians(TILT))
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
    ap.add_argument("--style", default="ribbon",
                    choices=["ribbon", "bars", "surface"],
                    help="ribbon: a wall along each street, continuous with "
                         "the line beneath it. bars: one upright per node. "
                         "surface: an interpolated sheet -- smooth, but it "
                         "reads as terrain rather than as streets")
    ap.add_argument("--verify", action="store_true",
                    help="check each surface's peak against the highest node")
    ap.add_argument("--context-km", type=float, default=1.6,
                    help="radius of the wider city plan drawn behind the "
                         "stack; 0 to omit")
    ap.add_argument("--relief", type=float, default=0.26,
                    help="height of the surface at full value, as a fraction "
                         "of the frame span")
    ap.add_argument("--grid", type=int, default=260,
                    help="interpolation grid, cells across the frame")
    ap.add_argument("--reach", type=float, default=35.0,
                    help="metres from a node beyond which the surface stops")
    ap.add_argument("--smooth", type=float, default=0.8,
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

    # the field, once: grid, interpolate, smooth, mask by distance.
    # Only the surface style needs it; ribbon and bars read the nodes directly.
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
    namecol = "osm_name" if "osm_name" in g.columns else "street_name"
    if args.style != "surface":
        print(f"style: {args.style}")
    VERIFY = {}
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
        X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
        t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) * args.relief * span

        if args.style in ("ribbon", "bars"):
            segs = np.c_[X, Y][pairs] if len(pairs) else np.zeros((0, 2, 2))
            ax.add_collection(LineCollection(segs, colors="#d0d0d0", lw=.6,
                                             zorder=k * 10 + 1))
            if args.style == "bars":
                # one upright per node, drawn far to near so a tall bar in
                # front covers a short one behind it
                up = np.stack([np.c_[X, Y], np.c_[X, Y + t]], axis=1)
                o = np.argsort(-Y)
                ax.add_collection(LineCollection(
                    up[o], cmap=cmap, array=v[o], lw=1.25,
                    norm=plt.Normalize(lo, hi), zorder=k * 10 + 2))
            else:
                # a wall along the street: neighbouring quads share an edge, so
                # the strip is continuous exactly as the street is
                a, b = pairs[:, 0], pairs[:, 1]
                quads = np.stack([
                    np.c_[X[a], Y[a]], np.c_[X[b], Y[b]],
                    np.c_[X[b], Y[b] + t[b]], np.c_[X[a], Y[a] + t[a]]], axis=1)
                val = np.nanmean(v[pairs], axis=1)
                o = np.argsort(-quads[:, :, 1].mean(axis=1))
                ax.add_collection(PolyCollection(
                    quads[o], cmap=cmap, array=val[o],
                    norm=plt.Normalize(lo, hi), linewidths=0,
                    zorder=k * 10 + 2))
                top = np.stack([np.c_[X[a], Y[a] + t[a]],
                                np.c_[X[b], Y[b] + t[b]]], axis=1)
                ax.add_collection(LineCollection(
                    top[o], cmap=cmap, array=val[o], lw=.85,
                    norm=plt.Normalize(lo, hi), zorder=k * 10 + 3))
            if args.verify:
                i = int(np.argmax(v))
                VERIFY[col] = (g[namecol].iloc[i], g[namecol].iloc[i])
            lx, ly = X.min() - span * 0.20, Y[np.argmin(X)]
            ax.text(lx, ly, label, color=INK, fontsize=8.2,
                    ha="right", va="center")
            ax.text(lx, ly - span * 0.05, f"median {g[col].median():.3f}",
                    color=MUT, fontsize=6.4, ha="right", va="center")
            continue

        X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
        lx, ly = X.min() - span * 0.20, Y[np.argmin(X)]
        ax.text(lx, ly, label, color=INK, fontsize=8.2, ha="right", va="center")
        ax.text(lx, ly - span * 0.05, f"median {g[col].median():.3f}",
                color=MUT, fontsize=6.4, ha="right", va="center")

    if args.verify:
        # The figure is read for where things are highest, so the cheap check
        # is whether the drawn peak sits on the street holding the highest
        # node. The cubic fit failed this on imageability and put the peak on
        # the wrong street entirely, which nothing else would have caught.
        print("\n  peak check:")
        print(f"    {'panel':<10}{'highest node is on':<26}{'surface peaks on':<26}")
        for _col, _, _, _ in LAYERS:
            if _col not in VERIFY:
                continue
            top, hit = VERIFY[_col]
            print(f"    {_col:<10}{str(top)[:24]:<26}{str(hit)[:24]:<26}"
                  f"{'ok' if top == hit else 'DIFFERS'}")

    # THE WINDOW IS THE STUDY AREA, not everything drawn. autoscale fits the
    # context plan too, and 1.6 km of network around a 1 km study area put the
    # stack in the middle third of the frame at a third of the size. The
    # context is meant to run off the edges; that is what makes it context.
    cx_all, cy_all = [], []
    for k in range(len(LAYERS)):
        X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
        cx_all.append(X); cy_all.append(Y)
    cx_all = np.concatenate(cx_all); cy_all = np.concatenate(cy_all)
    mx = (cx_all.max() - cx_all.min()) * 0.06
    # room on the left for the stratum labels, and above for the top relief
    ax.set_xlim(cx_all.min() - span * 0.30, cx_all.max() + mx)
    ax.set_ylim(cy_all.min() - mx, cy_all.max() + args.relief * span * 1.25 + mx)
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
