"""Four maps in the order of the equation: SIM, then G, M, P.

Each dimension carries the colour it has in the equation -- green for
habitat, amber for morphological, brick for permeability -- so a reader can
move between the formula and the map without a legend lookup. SIM keeps
magma because it is the composite and should not read as any one dimension.

Ramps run from the page background up to the accent rather than using a
matplotlib default: on a dark ground a standard sequential map puts its low
end at near-white, which inverts the reading.

Values are interpolated to 1 m ALONG each chain and never across one. A
holdout test puts linear interpolation at R2 = 0.84 for predicting an unseen
midpoint against 0.76 for nearest-node, so the ribbon is a reconstruction of
a continuous signal rather than decoration -- but interpolated points are a
deterministic function of measured ones and must never be treated as data.

    .venv/Scripts/python tools/sim_maps.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import cmcrameri.cm as cmc          # Crameri scientific colour maps
import cmocean                      # cmocean, built for ocean data

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

BG = "#0d1117"
FG = "#e6edf3"
DIM = "#8b949e"
GAP_M = 40                      # never bridge a real coverage gap

# Four distinct perceptually-uniform ramps rather than four tints of one
# accent. A dark-to-accent gradient carries most of its range in luminance
# alone, so a variable with a narrow spread reads as one flat colour; these
# move through hue as well and separate values that differ slightly.
# viridis suits G because its mid-range is green, which keeps the panel
# legible as the greenery map without inventing a scale.
# Direction matters as much as hue: on a dark ground the high end must be the
# bright one, or the strongest blocks sink into the page. lajolla_r and ice_r
# run light-to-dark and are the wrong way round here.
#
# Four hue families that do not collide. SIM keeps magma, which travels
# purple -> red -> orange -> cream; that range is why a red ramp for P read
# as a second copy of it, and why cividis read as neither amber nor
# anything else in particular. M is gold-dominant and P is teal, which is
# the one warm/cool axis magma does not occupy.
# Direction matters as much as hue: on a dark ground the high end must be the
# bright one, or the strongest blocks sink into the page. lajolla_r and ice_r
# run light-to-dark and are the wrong way round here.
#
# Four hue families that do not collide. SIM keeps magma, which travels
# purple -> red -> orange -> cream; that range is why a red ramp for P read
# as a second copy of it. lajolla and ice are perceptually uniform and
# colour-vision-deficiency safe, which magma and viridis already are, so the
# whole set holds together for a reader who cannot separate red from green.
PANELS = [
    ("SIM", "magma",         "SIM", "the composite"),
    ("G",   "viridis",       "G",   "green / habitat"),
    ("M",   cmc.lajolla,     "M",   "morphological"),
    ("P",   cmocean.cm.ice,  "P",   "permeability"),
]


def ramp(spec, name):
    if isinstance(spec, (str,)) or hasattr(spec, "__call__"):
        return spec
    return LinearSegmentedColormap.from_list(name, spec)


def ribbon(ax, d, col, cmap, lo, hi):
    segs, vals = [], []
    for _, g in d.groupby("chain"):
        g = g.sort_values("chain_pos_m")
        P = np.c_[g.geometry.x, g.geometry.y]
        pos, y = g.chain_pos_m.values, g[col].values
        for k in range(len(g) - 1):
            if pos[k + 1] - pos[k] > GAP_M:
                continue
            t = np.linspace(0, 1, 12)
            pts = P[k] + np.outer(t, P[k + 1] - P[k])
            vv = y[k] + t * (y[k + 1] - y[k])
            for j in range(len(t) - 1):
                segs.append([pts[j], pts[j + 1]])
                vals.append((vv[j] + vv[j + 1]) / 2)
    lc = LineCollection(segs, cmap=cmap, linewidths=5.2, array=np.array(vals),
                        norm=plt.Normalize(lo, hi))
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def main():
    banner("SIM dimension maps")
    m = gpd.read_file(PROC / "metrics.gpkg")
    d = m.merge(pd.read_csv(PROC / "sim_index.csv"), on="node_id")
    if "in_study" in d.columns:
        d = d[d.in_study]
    print(f"nodes: {len(d)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.6), facecolor=BG,
                             gridspec_kw=dict(hspace=.10, wspace=.02))
    for ax, (col, spec, sym, label) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(BG)
        # Clip the ramp to the 2nd-98th percentile. Across the full range a
        # handful of extreme nodes flatten everything else to one tone, and
        # the point of these panels is where a dimension varies.
        lo, hi = d[col].quantile([.02, .98])
        lc = ribbon(ax, d, col, ramp(spec, col), lo, hi)
        cb = fig.colorbar(lc, ax=ax, shrink=.62, pad=.015, aspect=26)
        cb.ax.yaxis.set_tick_params(color=FG, labelsize=8)
        cb.outline.set_edgecolor(DIM)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
        for st, g in d.groupby("osm_name"):
            i = g.geometry.y.idxmax()
            ax.annotate(st.replace(" Street", "").replace(" Avenue", " Ave"),
                        (g.loc[i, "geometry"].x, g.loc[i, "geometry"].y),
                        fontsize=6, color=DIM, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        med = d[col].median()
        ax.set_title(f"{sym}   {label}", color=FG, fontsize=13.5,
                     loc="left", pad=20, fontweight="semibold")
        ax.text(0, 1.006,
                f"median {med:.3f}    full range {d[col].min():.3f}–{d[col].max():.3f}"
                f"    ramp clipped to {lo:.3f}–{hi:.3f}",
                transform=ax.transAxes, color=DIM, fontsize=8, va="bottom")

    fig.suptitle("SIM  =  0.34 G  +  0.33 M  +  0.33 P", color=FG,
                 fontsize=17, y=.985)
    out = RES / "figures" / "sim_dwell_map.png"
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor=BG)
    print(f"wrote {out}")
    print(d[["G", "M", "P", "SIM"]].describe().loc[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
