"""The three SIM terms, one row per city.

M is drawn elsewhere -- tools/m_maps.py gives it a panel each and the room to
be read. This is the other half of that split: I, Y and D side by side, so the
question it answers is which TERM separates the two cities, not how the
composite lands.

ONE SCALE PER COLUMN, shared down it. A colour means the same value in Murray
Hill as in the City of London, which is the comparison; the three terms do not
share a scale with each other, because they are not the same quantity and a
common one would only say that imageability is smaller than identity.

I_raw AND D_raw, not I and D. The sigmoid-transformed versions depend on tau,
so a cross-city panel of them is a picture of the calibration choice as much as
of the streets. The raw terms carry no tau, so the comparison stands whichever
calibration the text quotes.

Rows are sized by each frame's own extent at a common metres-per-inch, so the
City of London reading as the smaller area is a fact about the study areas and
not about the layout.

    .venv/Scripts/python tools/sim_terms_maps.py
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

import cmcrameri.cm as cmc
import cmocean

BG, FG, MUT = "#0e0f12", "#e8e6e1", "#9a9aa2"

TERMS = [("I_raw", "place imageability", "viridis"),
         ("Y",     "place identity",     cmc.lajolla),
         ("D_raw", "place dependence",   cmocean.cm.ice)]
CITIES = [("Murray Hill, Manhattan", "data/processed/nodes.gpkg",
           "results/tables/vlm_calculations.csv", 32618),
          ("City of London", "data/london/processed/nodes.gpkg",
           "results/london/tables/vlm_calculations.csv", 27700)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/sim_terms_two_cities.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--slide", default="16:9", choices=["16:9", "4:3", "free"],
                    help="size the canvas to a PowerPoint page. 16:9 is "
                         "13.333x7.5 in, 4:3 is 10x7.5, free is 13.6x9.6")
    args = ap.parse_args()
    banner("the three terms, one row per city")

    frames = []
    for name, gpkg, calc, crs in CITIES:
        n = gpd.read_file(gpkg)[["node_id", "geometry"]].to_crs(crs)
        c = pd.read_csv(calc)
        per = c.groupby("node_id")[[t[0] for t in TERMS]].mean().reset_index()
        g = n.merge(per, on="node_id", how="inner")
        g["x"], g["y"] = g.geometry.x, g.geometry.y
        frames.append((name, g))
        print(f"  {name:<26}{len(g):>5} nodes")

    spans = [max(np.ptp(g.x.values), np.ptp(g.y.values)) for _, g in frames]
    S = max(spans)
    # one normalisation per term, pooled over both cities
    norms = {}
    for col, _, _ in TERMS:
        v = np.concatenate([g[col].dropna().values for _, g in frames])
        norms[col] = np.percentile(v, [2, 98])
        print(f"  {col:<8}scale {norms[col][0]:.3f} to {norms[col][1]:.3f}")

    # A slide is wide and short, and six square-ish map panels are the
    # opposite, so the panels are what has to give: the margins tighten and the
    # colourbars move up under the bottom row rather than sitting in a band of
    # their own. Filling the page exactly means no rescaling in PowerPoint,
    # which is where figure text usually goes soft.
    SIZE = {"16:9": (13.333, 7.5), "4:3": (10.0, 7.5), "free": (13.6, 9.6)}
    fig = plt.figure(figsize=SIZE[args.slide], facecolor=BG)
    tight = args.slide != "free"
    gs = fig.add_gridspec(2, 3, height_ratios=[s / S for s in spans],
                          hspace=.05 if tight else .07,
                          wspace=.025 if tight else .03,
                          left=.05, right=.99,
                          top=.915 if tight else .93,
                          bottom=.135 if tight else .115)

    for r, (name, g) in enumerate(frames):
        for c, (col, label, cmap) in enumerate(TERMS):
            ax = fig.add_subplot(gs[r, c], facecolor=BG)
            lo, hi = norms[col]
            ax.scatter(g.x, g.y, c=g[col], cmap=cmap, vmin=lo, vmax=hi,
                       s=8.5, linewidths=0)
            ax.set_aspect("equal")
            cx, cy = g.x.mean(), g.y.mean()
            h = S / 2 * 1.05
            foot = h * 0.20 if r == len(frames) - 1 else 0.0
            ax.set_xlim(cx - h, cx + h)
            ax.set_ylim(cy - h - foot, cy + h)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#23262b")
            if r == 0:
                ax.set_title(label, color=FG, fontsize=12, pad=8)
            if c == 0:
                ax.text(-.03, .5, name, transform=ax.transAxes, color=FG,
                        fontsize=11.5, rotation=90, va="center", ha="right")
            ax.text(.025, .97, f"median {g[col].median():.3f}",
                    transform=ax.transAxes, color=MUT, fontsize=8, va="top")
            # scale bar under the bottom row only, label below the bar so it
            # grows away from the map rather than back into it
            if r == len(frames) - 1 and c == 0:
                bar, by = 200.0, cy - h - foot * 0.40
                bx = cx - h * 0.92
                ax.plot([bx, bx + bar], [by, by], color=FG, lw=1.8)
                ax.text(bx + bar / 2, by - foot * 0.13, "200 m", color=FG,
                        ha="center", va="top", fontsize=8)

    # one colourbar per column, under it, so the scale sits with the term
    for c, (col, _, cmap) in enumerate(TERMS):
        box = gs[1, c].get_position(fig)
        cax = fig.add_axes([box.x0 + box.width * .22,
                            .072 if tight else .062,
                            box.width * .56, .014 if tight else .012])
        lo, hi = norms[col]
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi),
                                                cmap=cmap),
                          cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7.5, colors=MUT, length=2)
        cb.outline.set_edgecolor("#23262b")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
