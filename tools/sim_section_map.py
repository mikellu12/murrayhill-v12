"""SIM scores aggregated to street sections, mapped and tabulated.

A section is one block-length run of a street between crossings. face_id is
the project's clustering unit but its groups run 1 to 47 nodes -- it is a
street side, not a block -- so it is too coarse to show where the index moves
along a street.

Sections are cut where the frame says a crossing is: a node within
crossing_m of a node on a different street. Those nodes are the ones whose
H/W is inherited rather than measured, for the same reason -- there is no
street wall on the perpendicular -- so the cut points are the same places the
geometry breaks down.

Every section carries the mean M of its half-views, both sides pooled. A
section with fewer than min_n scored half-views is drawn but not labelled;
its mean is one or two observations and the map would imply more.

    .venv/Scripts/python tools/sim_section_map.py
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import sim_cmap, PROJ_CRS, PROC, RAW, RES, banner

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
CMAP = sim_cmap()          # the shared ramp; see common.SIM_CMAP_STOPS


def sections(m, crossing_m):
    """Cut each street into block-length runs at its crossings."""
    xy = np.c_[m.geometry.x, m.geometry.y]
    nm = m.osm_name.to_numpy()
    at = np.zeros(len(m), bool)
    for i in range(len(m)):
        o = nm != nm[i]
        if o.any():
            at[i] = np.hypot(xy[o, 0] - xy[i, 0], xy[o, 1] - xy[i, 1]).min() < crossing_m
    m = m.assign(at_crossing=at)

    out = np.full(len(m), -1)
    sec = 0
    for street, g in m.groupby("osm_name"):
        idx = g.index.to_numpy()
        p = xy[m.index.get_indexer(idx)]
        # order along the street's own principal axis
        c = p - p.mean(0)
        v = np.linalg.svd(c, full_matrices=False)[2][0]
        order = idx[np.argsort(c @ v)]
        prev_at = True
        for nid in order:
            k = m.index.get_loc(nid)
            if m.at_crossing.iloc[k]:
                out[k] = -1
                prev_at = True
                continue
            if prev_at:
                sec += 1
            out[k] = sec
            prev_at = False
    return m.assign(section=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossing-m", type=float, default=15.0)
    ap.add_argument("--min-n", type=int, default=4)
    args = ap.parse_args()
    banner("SIM by street section")

    m = gpd.read_file(PROC / "metrics.gpkg").to_crs(UTM).reset_index(drop=True)
    sc = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    keep = set(sc.node_id)
    m = m[m.node_id.isin(keep)].reset_index(drop=True)
    m = sections(m, args.crossing_m)

    node_M = sc.groupby("node_id").M.mean().rename("M")
    m = m.merge(node_M, on="node_id", how="left")

    lab = m[m.section > 0]
    agg = (lab.groupby("section")
              .agg(street=("osm_name", "first"), n=("M", "count"),
                   M=("M", "mean"), sd=("M", "std"),
                   e=("geometry", lambda g: g.x.mean()),
                   n_=("geometry", lambda g: g.y.mean()))
              .reset_index())
    agg = agg[agg.n > 0]
    print(f"{len(m)} nodes -> {len(agg)} sections "
          f"({int((m.section < 0).sum())} at crossings, unassigned)")
    print(f"section size: median {agg.n.median():.0f} nodes, "
          f"range {agg.n.min():.0f}-{agg.n.max():.0f}\n")

    show = agg[agg.n >= args.min_n].sort_values("M")
    print(f"  {'street':<22}{'n':>4}{'mean M':>9}{'sd':>7}")
    for r in show.head(6).itertuples():
        print(f"  {r.street:<22}{r.n:>4}{r.M:>9.3f}{r.sd:>7.3f}")
    print(f"  {'...':<22}")
    for r in show.tail(6).itertuples():
        print(f"  {r.street:<22}{r.n:>4}{r.M:>9.3f}{r.sd:>7.3f}")

    bf = gpd.read_file(RAW / "building_footprints.geojson").to_crs(UTM)
    vmin, vmax = np.nanpercentile(agg.M, [5, 95])
    norm = Normalize(vmin, vmax)

    fig, ax = plt.subplots(figsize=(14, 12))
    bf.plot(ax=ax, color="#EDF0EE", edgecolor="#DCE2DE", linewidth=0.35, zorder=1)
    m[m.section < 0].plot(ax=ax, color="#C8CFCA", markersize=9, zorder=2)
    sm = m[m.section > 0].merge(agg[["section", "M"]].rename(
        columns={"M": "sec_M"}), on="section", how="left")
    ax.scatter(sm.geometry.x, sm.geometry.y, c=sm.sec_M, cmap=CMAP, norm=norm,
               s=26, edgecolor="white", linewidth=0.4, zorder=3)

    for r in agg[agg.n >= args.min_n].itertuples():
        ax.annotate(f"{r.M:.2f}", (r.e, r.n_), fontsize=7.5, ha="center",
                    va="center", zorder=5, color="#12211C",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#CFD8D2", lw=0.5, alpha=0.93))
    ax.set_axis_off()
    ax.set_title(f"SIM by street section\n{len(agg)} sections, "
                 f"{int(agg.n.sum())} nodes, mean M {agg.M.mean():.3f}",
                 fontsize=13, loc="left", pad=14)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                      fraction=0.028, pad=0.01)
    cb.set_label("mean M", fontsize=10)
    cb.outline.set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", color="none",
                              markerfacecolor="#C8CFCA", markersize=8,
                              label="at a crossing, no section")],
              loc="lower left", frameon=False, fontsize=9)

    out = RES / "figures" / "sim_section_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    agg.drop(columns=["e", "n_"]).to_csv(
        RES / "tables" / "vlm_sections.csv", index=False)
    print(f"\nwrote {out}")
    print(f"wrote {RES / 'tables' / 'vlm_sections.csv'}")


if __name__ == "__main__":
    main()
