"""Four maps for the VLM index: M, then I_raw, Y, D_raw.

The pixel-driven counterpart is tools/sim_maps.py, which draws SIM, G, M, P.
This keeps that figure's grammar exactly -- same ribbons, same dark ground,
same ramp per role -- so the two sets can be read side by side without
relearning the colours. The composite keeps magma; the green dimension keeps
viridis; the morphological one keeps lajolla; the permeability one keeps ice.

I_raw and D_raw are drawn, not I and D. Three quarters of D sits above 0.95
because tau_D was calibrated for pixel shares rather than normalised ratings,
so a D panel would be one flat tone and would map the threshold rather than
the street.

Each node pools its four half-views. That is the right unit here: the ribbon
runs along a chain, and a chain is a line of nodes, not of half-views.

Values are interpolated to 1 m ALONG each chain and never across one, and the
ribbon breaks at any gap over GAP_M so a coverage hole never reads as a
smooth transition. Interpolated points are a deterministic function of
measured ones and are not data.

    .venv/Scripts/python tools/sim_vlm_maps.py
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
import cmcrameri.cm as cmc
import cmocean

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

BG = "#0d1117"
FG = "#e6edf3"
DIM = "#8b949e"
GAP_M = 40

PANELS = [
    ("M",     "magma",        "M",     "the composite"),
    ("I_raw", "viridis",      "I_raw", "place imageability"),
    ("Y",     cmc.lajolla,    "Y",     "place identity"),
    ("D_raw", cmocean.cm.ice, "D_raw", "place dependence"),
]


def ribbon(ax, d, col, cmap, lo, hi):
    segs, vals = [], []
    for _, g in d.groupby("chain"):
        g = g.sort_values("chain_pos_m")
        P = np.c_[g.geometry.x, g.geometry.y]
        pos, y = g.chain_pos_m.values, g[col].values
        for k in range(len(g) - 1):
            if pos[k + 1] - pos[k] > GAP_M or not np.isfinite([y[k], y[k + 1]]).all():
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
    banner("VLM index maps")
    m = gpd.read_file(PROC / "metrics.gpkg")
    v = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    node = v.groupby("node_id")[["I_raw", "Y", "D_raw", "M"]].mean().reset_index()
    d = m.merge(node, on="node_id", how="inner")
    print(f"{len(v)} half-views -> {len(node)} nodes -> {len(d)} with geometry")
    print(f"M present on {int(d.M.notna().sum())} of them\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.6), facecolor=BG,
                             gridspec_kw=dict(hspace=.10, wspace=.02))
    for ax, (col, cmap, sym, label) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(BG)
        # 2nd-98th percentile: across the full range a few extreme nodes
        # flatten everything else to one tone, and these panels exist to show
        # where a dimension varies.
        lo, hi = d[col].quantile([.02, .98])
        lc = ribbon(ax, d, col, cmap, lo, hi)
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
        s = d[col].dropna()
        ax.set_title(f"{sym}   {label}", color=FG, fontsize=13.5,
                     loc="left", pad=20, fontweight="semibold")
        ax.text(0, 1.006,
                f"median {s.median():.3f}    full range {s.min():.3f}–{s.max():.3f}"
                f"    ramp clipped to {lo:.3f}–{hi:.3f}"
                + (f"    {int(d[col].isna().sum())} nodes without a value"
                   if d[col].isna().any() else ""),
                transform=ax.transAxes, color=DIM, fontsize=8, va="bottom")

    fig.suptitle("M  =  I^a · Y^b · D^c · Ω        drawn from the VLM ratings",
                 color=FG, fontsize=17, y=.985)
    out = RES / "figures" / "sim_vlm_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
