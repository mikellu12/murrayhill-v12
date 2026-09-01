"""M mapped for both study areas on ONE colour scale.

A per-city colour scale makes every city look the same: the darkest street in
each is the darkest colour, whatever value it holds. The comparison the paper
wants is between cities, so both panels share one normalisation, and the
legend says what the shared range is.

A_i IS OFF IN BOTH. Murray Hill can compute the canyon penalty and London,
having no building heights, cannot. Comparing a city that carries the term
against one that does not compares the presence of the term, not the streets:
Murray Hill's median moves 0.472 to 0.630 when it is dropped. M_noA is the
like-for-like column and it is what is drawn.

The two frames are separately projected and thousands of kilometres apart, so
they are drawn as two panels at a common metre-per-pixel scale rather than on
one pair of axes, with a scale bar so the reader can see that the City of
London frame is the smaller area.

    .venv/Scripts/python tools/m_maps.py
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

BG = "#0e0f12"
FG = "#e8e6e1"
# magma, matching the M panel of tools/sim_vlm_maps.py and the M row of
# tools/sim_maps.py. M is drawn in three places and should not change colour
# between them.
CMAP = "magma"


def load(nodes_gpkg, calc_csv, crs, m_col):
    n = gpd.read_file(nodes_gpkg)[["node_id", "geometry"]].to_crs(crs)
    c = pd.read_csv(calc_csv)
    col = m_col if m_col in c.columns else "M"
    per = c.groupby("node_id")[col].mean().rename("M").reset_index()
    g = n.merge(per, on="node_id", how="inner").dropna(subset=["M"])
    g["x"], g["y"] = g.geometry.x, g.geometry.y
    return g, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/m_two_cities.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", action="store_true",
                    help="draw the figure heading; off by default so the "
                         "figure can carry a caption of its own")
    args = ap.parse_args()
    banner("M for both cities, one colour scale")

    panels = []
    for name, gpkg, calc, crs, col in (
            ("Murray Hill, Manhattan", "data/processed/nodes.gpkg",
             "results/tables/vlm_calculations.csv", 32618, "M_noA"),
            ("City of London", "data/london/processed/nodes.gpkg",
             "results/london/tables/vlm_calculations.csv", 27700, "M")):
        g, used = load(Path(gpkg), Path(calc), crs, col)
        panels.append((name, g, used))
        print(f"  {name:<26}{len(g):>5} nodes   column {used:<7}"
              f"median {g.M.median():.3f}")

    allM = np.concatenate([g.M.values for _, g, _ in panels])
    vmin, vmax = np.percentile(allM, [2, 98])
    print(f"\n  shared scale {vmin:.3f} to {vmax:.3f} "
          f"(2nd-98th percentile of both cities pooled)")

    # common metres-per-inch so the two frames are comparable in size
    spans = [max(np.ptp(g.x.values), np.ptp(g.y.values)) for _, g, _ in panels]
    # The heading gets a reserved band measured in inches. Placing figure text
    # at fixed fractions while the panel titles are placed in points below the
    # axes puts the two on different measures, and they collide as soon as the
    # figure changes size.
    fig = plt.figure(figsize=(15.5, 8.2), facecolor=BG)
    head = 0.16 / 8.2      # panel titles only; no figure heading
    gs = fig.add_gridspec(1, 2, width_ratios=[s / max(spans) for s in spans],
                          wspace=0.06, left=.03, right=.88,
                          top=1 - head - 0.045, bottom=.08)
    sc = None
    for i, (name, g, used) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], facecolor=BG)
        sc = ax.scatter(g.x, g.y, c=g.M, cmap=CMAP, vmin=vmin, vmax=vmax,
                        s=17, linewidths=0)
        ax.set_aspect("equal")
        cx, cy = g.x.mean(), g.y.mean()
        h = max(spans) / 2 * 1.06
        ax.set_xlim(cx - h, cx + h)
        ax.set_ylim(cy - h, cy + h)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a2d33")
        ax.set_title(f"{name}\n{len(g)} nodes   median M {g.M.median():.3f}"
                     f"   ({used})",
                     color=FG, fontsize=12.5, pad=12)
        bar = 200.0
        x0, y0 = cx - h * 0.92, cy - h * 0.92
        ax.plot([x0, x0 + bar], [y0, y0], color=FG, lw=2.2)
        ax.text(x0 + bar / 2, y0 + h * 0.03, "200 m", color=FG,
                ha="center", fontsize=9)

        # North. Both frames are drawn in a projected CRS whose grid north runs
        # straight up the page, so the arrow is vertical in each panel -- it is
        # GRID north, which differs from true north by the meridian convergence
        # of the projection, well under a degree in both study areas.
        ax_ = cx + h * 0.80
        ay = cy - h * 0.86
        arrow = h * 0.13
        ax.annotate("", xy=(ax_, ay + arrow), xytext=(ax_, ay),
                    arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.4,
                                    mutation_scale=13))
        ax.text(ax_, ay + arrow + h * 0.035, "N", color=FG, ha="center",
                va="bottom", fontsize=10.5)

    if args.title:
        fig.text(.03, 1 - head * 0.36, "M across two study areas, "
                 "one shared colour scale", color=FG, fontsize=16.5, va="center")

    cax = fig.add_axes([0.90, 0.14, 0.016, 0.66])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("M", color=FG, fontsize=11)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=FG)
    cb.outline.set_edgecolor("#2a2d33")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
