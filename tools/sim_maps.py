"""I, Y, D and M mapped for both study areas, one colour scale per dimension.

Four dimensions, two cities, eight panels. The scale is shared ACROSS CITIES
within a row and free BETWEEN rows: a colour means the same value in Murray
Hill as in London, which is the comparison, but I and D and M do not share a
scale with each other because they are not the same quantity.

A_i IS OFF IN BOTH. London has no building heights and cannot compute the
canyon penalty at all, so M_noA is the like-for-like column; Murray Hill's
median is 0.630 with the penalty and 0.661 without, and reading one city's M
against the other's M_noA compares the presence of the term rather than the
streets.

GLOBAL TAU, NOT LOCAL. Local tau recentres each city's sigmoid on its own
median, which puts every city at 0.5 by construction and makes a cross-city
panel meaningless. The local columns are for ranking streets inside one city.

    .venv/Scripts/python tools/sim_maps.py
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner

BG, FG, MUT = "#0e0f12", "#e8e6e1", "#9a9aa2"
# The ramps are the ones picked for tools/sim_vlm_maps.py, one per dimension
# and each from a different scheme. A shared ramp was the wrong instinct: these
# four panels are not four readings of one quantity, and giving each dimension
# its own identity is what lets a reader hold all four in mind at once. Reused
# rather than re-picked, so this figure and the single-city one agree.
#
# I_raw and D_raw, not I and D. The sigmoid-transformed versions depend on tau,
# so they would make a cross-city panel a picture of the calibration choice as
# much as of the streets; the raw dimensions carry no tau at all.
import cmcrameri.cm as cmc
import cmocean
DIMS = [("I_raw", "place imageability", "viridis"),
        ("Y", "place identity", cmc.lajolla),
        ("D_raw", "place dependence", cmocean.cm.ice),
        ("M", "the composite", "magma")]
CITIES = [("Murray Hill, Manhattan", "data/processed/nodes.gpkg",
           "results/tables/vlm_calculations.csv", 32618, "M_noA"),
          ("City of London", "data/london/processed/nodes.gpkg",
           "results/london/tables/vlm_calculations.csv", 27700, "M")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/sim_maps_two_cities.png"))
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    banner("I, Y, D, M for both cities")

    frames = []
    for name, gpkg, calc, crs, mcol in CITIES:
        n = gpd.read_file(gpkg)[["node_id", "geometry"]].to_crs(crs)
        c = pd.read_csv(calc)
        use = mcol if mcol in c.columns else "M"
        cols = {d: (use if d == "M" else d) for d, _, _ in DIMS}
        per = c.groupby("node_id")[list(set(cols.values()))].mean().reset_index()
        g = n.merge(per, on="node_id", how="inner")
        g["x"], g["y"] = g.geometry.x, g.geometry.y
        frames.append((name, g, cols, use))
        print(f"  {name:<26}{len(g):>5} nodes, M column {use}")

    spans = [max(np.ptp(g.x.values), np.ptp(g.y.values)) for _, g, _, _ in frames]
    S = max(spans)

    # The heading block needs its own space rather than being written over the
    # top row: two lines of figure text at .975/.955 plus a column title with
    # padding collided at top=.945, and the collision grows with figure height
    # because the text is placed in figure fractions and the title in points.
    fig = plt.figure(figsize=(11.4, 4.15 * len(DIMS)), facecolor=BG)
    head = 0.44 / (4.15 * len(DIMS))          # inches of heading, as a fraction
    gs = fig.add_gridspec(len(DIMS), 2,
                          width_ratios=[s / S for s in spans],
                          hspace=.10, wspace=.04,
                          left=.045, right=.86, top=1 - head - 0.035, bottom=.02)
    for r, (d, label, cmap) in enumerate(DIMS):
        vals = np.concatenate([g[c[d]].dropna().values for _, g, c, _ in frames])
        vmin, vmax = np.percentile(vals, [2, 98])
        sc = None
        for k, (name, g, cols, _) in enumerate(frames):
            ax = fig.add_subplot(gs[r, k], facecolor=BG)
            v = g[cols[d]]
            sc = ax.scatter(g.x, g.y, c=v, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=9, linewidths=0)
            ax.set_aspect("equal")
            cx, cy = g.x.mean(), g.y.mean()
            h = S / 2 * 1.06
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_color("#23262b")
            if r == 0:
                ax.set_title(name, color=FG, fontsize=12, pad=6)
            ax.text(.02, .965, f"median {v.median():.3f}", transform=ax.transAxes,
                    color=MUT, fontsize=8.5, va="top")
            # north and scale, on the bottom row only: repeating them on every
            # row would say they change between rows, and they do not
            if r == len(DIMS) - 1:
                bar = 200.0
                bx, by = cx - h * 0.92, cy - h * 0.90
                ax.plot([bx, bx + bar], [by, by], color=FG, lw=1.8)
                ax.text(bx + bar / 2, by + h * 0.035, "200 m", color=FG,
                        ha="center", fontsize=7.5)
                nx, ny, arrow = cx + h * 0.82, cy - h * 0.88, h * 0.12
                ax.annotate("", xy=(nx, ny + arrow), xytext=(nx, ny),
                            arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.2,
                                            mutation_scale=11))
                ax.text(nx, ny + arrow + h * 0.03, "N", color=FG, ha="center",
                        va="bottom", fontsize=9)
            if k == 0:
                ax.text(-.035, .5, label, transform=ax.transAxes, color=FG,
                        fontsize=11, rotation=90, va="center", ha="right")
        box = gs[r, 1].get_position(fig)
        cax = fig.add_axes([.875, box.y0 + box.height * .16, .015,
                            box.height * .68])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.yaxis.set_tick_params(color=FG, labelsize=8)
        plt.setp(plt.getp(cb.ax, "yticklabels"), color=FG)
        cb.outline.set_edgecolor("#23262b")

    fig.text(.045, 1 - head * 0.34, "The three dimensions and M, both study areas",
             color=FG, fontsize=15, va="center")
    fig.text(.045, 1 - head * 0.78,
             "Node means. Each row shares one colour scale across the two "
             "cities; rows do not share with each other. Canyon penalty off in "
             "both, global tau.", color=MUT, fontsize=9, va="center")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
