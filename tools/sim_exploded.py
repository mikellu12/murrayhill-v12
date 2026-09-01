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

RED IS THE UPPER TAIL, on every plane, at the same quantile. It is the one
thing the eye should be able to follow between strata, so it is not a ramp
value but a fixed mark: this node is in the top decile of THIS dimension. A
column with red at every level is a street that does well on all three; red at
the top and nowhere below is a composite carried by a single term.

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
from matplotlib.collections import LineCollection

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
    ap.add_argument("--gap", type=float, default=0.30,
                    help="layer separation, as a fraction of the frame's span")
    ap.add_argument("--tail", type=float, default=0.90,
                    help="quantile marked red on every plane")
    ap.add_argument("--dpi", type=int, default=260)
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
    g["_x"] -= g._x.mean(); g["_y"] -= g._y.mean()
    g = g.reset_index(drop=True)
    g["_i"] = np.arange(len(g))
    print(f"{len(g)} nodes with a score, M column {mcol}")

    pairs = chains(g)
    print(f"{len(pairs)} street segments")
    span = max(np.ptp(g._x.values), np.ptp(g._y.values))
    gap = args.gap * span

    fig, ax = plt.subplots(figsize=(9.2, 13.4), facecolor=BG)
    ax.set_facecolor(BG); ax.set_axis_off()

    # threads first, behind everything: one per node, tying the strata together
    thin = g.sample(min(len(g), 260), random_state=3)
    for _, r in thin.iterrows():
        xs, ys = [], []
        for k in range(len(LAYERS)):
            X, Y = iso(r._x, r._y, k, gap)
            xs.append(X); ys.append(Y)
        ax.plot(xs, ys, color="#d7d7d7", lw=.35, zorder=0, solid_capstyle="butt")

    for k, (col, label, cmap, short) in enumerate(LAYERS):
        X, Y = iso(g._x.to_numpy(), g._y.to_numpy(), k, gap)
        P = np.c_[X, Y]
        segs = P[pairs] if len(pairs) else np.zeros((0, 2, 2))
        if col == "fabric":
            ax.add_collection(LineCollection(segs, colors="#c4c4c4", lw=.9,
                                             zorder=k * 10 + 1))
        else:
            v = g[col].to_numpy()
            vals = np.nanmean(v[pairs], axis=1)
            lo, hi = np.nanpercentile(v, [3, 97])
            ax.add_collection(LineCollection(
                segs, cmap=cmap, array=vals, lw=2.3,
                norm=plt.Normalize(lo, hi), zorder=k * 10 + 1))
            tail = v >= np.nanquantile(v, args.tail)
            ax.scatter(X[tail], Y[tail], s=5.5, c=RED, linewidths=0,
                       zorder=k * 10 + 2)
        # label to the left, in the reference's manner
        lx = X.min() - span * 0.20
        ly = Y[np.argmin(X)]
        ax.text(lx, ly, label, color=INK if col != "fabric" else MUT,
                fontsize=8.2, ha="right", va="center", family="DejaVu Sans")
        if short:
            q = g[col]
            ax.text(lx, ly - span * 0.045,
                    f"median {q.median():.3f}", color=MUT, fontsize=6.4,
                    ha="right", va="center")

    ax.autoscale_view()
    ax.set_aspect("equal")
    fig.text(.06, .965, name, color=INK, fontsize=17)
    fig.text(.06, .945,
             f"M = I^a · Y^b · D^c, drawn as strata.  {len(g)} nodes.  "
             f"Red marks the top {(1-args.tail)*100:.0f}% of each dimension.",
             color=MUT, fontsize=8.6)
    out = args.out or RES / "figures" / f"sim_exploded_{CFG.get('study_area_slug','area')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
