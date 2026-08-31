"""The three SIM dimensions plotted against each other, by typology.

x = I_raw, y = Y, z = D_raw -- the composed dimensions BEFORE the sigmoids.
Deliberately the raw ones: D saturates at 0.99 for three quarters of the
frame, so a plot of D would be a plane at the ceiling and would say more about
tau_D than about Murray Hill.

Two figures, because a 3D scatter cannot be read on its own -- depth is
ambiguous, near points occlude far ones, and no projection is honest about
all three axes at once. The cube is what the manuscript's figure looks like;
the three pairwise panels are where you can actually see whether the
typologies separate.

Colour AND marker shape both carry typology. The palette passes the six-check
validation, but two of its pairs sit in the 6-8 CVD band, which is legal only
with a second encoding.

    .venv/Scripts/python tools/sim_cube.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

# validated: lightness band, chroma floor, CVD separation, contrast
STYLE = {
    "avenue_canyon":    ("#B5512F", "o", "Avenue canyons"),
    "mid_block":        ("#31805A", "^", "Mid-blocks"),
    "avenue_secondary": ("#C99A2E", "s", "Secondary avenues"),
    "other":            ("#6E5CA6", "D", "Other / porous breaks"),
}
AX = {"I_raw": "Place Imageability  (I_raw)",
      "Y": "Place Identity  (Y)",
      "D_raw": "Place Dependence  (D_raw)"}


def main():
    banner("the three dimensions, by typology")
    d = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "typology"]]
    # one point per node: the four half-views of a node are the same street
    n = (d.groupby("node_id")[["I_raw", "Y", "D_raw"]].mean()
           .reset_index().merge(met, on="node_id", how="left").dropna())
    print(f"{len(n)} nodes, {len(d)} half-views pooled four to a node\n")
    print(f"  {'typology':<20}{'n':>5}{'I_raw':>9}{'Y':>9}{'D_raw':>9}")
    for t, g in n.groupby("typology"):
        print(f"  {t:<20}{len(g):>5}{g.I_raw.mean():>9.3f}"
              f"{g.Y.mean():>9.3f}{g.D_raw.mean():>9.3f}")

    order = [t for t in STYLE if t in set(n.typology)]
    handles = [Line2D([], [], marker=STYLE[t][1], color="none",
                      markerfacecolor=STYLE[t][0], markeredgecolor="white",
                      markersize=9, label=STYLE[t][2]) for t in order]

    # ---- the cube -------------------------------------------------------
    fig = plt.figure(figsize=(11.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")
    for t in order:
        g = n[n.typology == t]
        c, mk, _ = STYLE[t]
        ax.scatter(g.I_raw, g.Y, g.D_raw, c=c, marker=mk, s=34, alpha=0.82,
                   edgecolor="white", linewidth=0.45, depthshade=False)
    ax.set_xlabel(AX["I_raw"], labelpad=12, fontsize=10)
    ax.set_ylabel(AX["Y"], labelpad=12, fontsize=10)
    ax.set_zlabel(AX["D_raw"], labelpad=8, fontsize=10)
    for s in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        s(0, 1)
    ax.view_init(elev=18, azim=-58)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(True, color="#D7DCD8", linewidth=0.6)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
    ax.set_title("The three SIM dimensions, one point per node\n"
                 "composed values, before the sigmoids",
                 fontsize=12.5, loc="left", pad=6)
    out = RES / "figures" / "sim_cube.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")

    # ---- the readable version -------------------------------------------
    pairs = [("I_raw", "Y"), ("I_raw", "D_raw"), ("Y", "D_raw")]
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for a, (xk, yk) in zip(axes, pairs):
        for t in order:
            g = n[n.typology == t]
            c, mk, _ = STYLE[t]
            a.scatter(g[xk], g[yk], c=c, marker=mk, s=30, alpha=0.75,
                      edgecolor="white", linewidth=0.4)
        a.set_xlabel(AX[xk], fontsize=10)
        a.set_ylabel(AX[yk], fontsize=10)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.grid(True, color="#E4E9E5", linewidth=0.7)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#C7CFC9")
    axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig2.suptitle("The same points, projected pairwise — where the separation "
                  "is actually legible", fontsize=12.5, x=0.09, ha="left")
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = RES / "figures" / "sim_cube_panels.png"
    fig2.savefig(out2, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
