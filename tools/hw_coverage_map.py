"""Where the unmeasured H/W nodes actually are.

501 of 733 nodes returned a facade-to-facade width from the cone probe. The
other 232 are not scattered noise: 87 per cent of nodes within 15 m of a
crossing fail, against a flat 12 per cent everywhere else, because at a
crossing the perpendicular points down the cross street and there is no
street wall on it to measure.

Draws every node over the building footprints, coloured by HW_source, so the
pattern is visible rather than asserted. Writes a figure and a per-street
table; changes no data.

    .venv/Scripts/python tools/hw_coverage_map.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, CFG, PROC, RAW, RES, banner

G = CFG["geometry"]
UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
COL = {"measured": "#1C5A49", "segment_median": "#C9932F", "none": "#9A4A31"}
LAB = {"measured": "measured from footprints",
       "segment_median": "inherited from the street segment",
       "none": "no value"}


def main():
    banner("H/W coverage across the frame")
    m = gpd.read_file(PROC / "metrics.gpkg").to_crs(UTM)
    if "HW_source" not in m.columns:
        sys.exit("run stage 5 first -- no HW_source column")
    bf = gpd.read_file(RAW / "building_footprints.geojson").to_crs(UTM)

    # metrics.csv predates the cleaned-frame join and carries every node s04
    # built, including the 54 Park Avenue nodes the cleaned frame excludes for
    # the Grand Central viaduct and tunnel. Only the export applies that flag,
    # so drawing metrics.csv shows 52 nodes the VLM never rated. Restrict to
    # what was actually rated, or the map answers a question nobody asked.
    rated = Path("results/tables/sim_vlm.csv")
    if rated.exists():
        keep = set(pd.read_csv(rated).node_id)
        before = len(m)
        m = m[m.node_id.isin(keep)].copy()
        print(f"restricted to the {len(m)} rated nodes "
              f"({before - len(m)} dropped: excluded upstream)")

    # distance to the nearest node on a different street, as a crossing proxy
    xy = np.c_[m.geometry.x, m.geometry.y]
    nm = m.osm_name.to_numpy()
    dx = []
    for i in range(len(m)):
        o = nm != nm[i]
        dx.append(np.hypot(xy[o, 0] - xy[i, 0], xy[o, 1] - xy[i, 1]).min()
                  if o.any() else np.nan)
    m["d_cross"] = dx

    miss = m[m.HW_source != "measured"]
    print(f"{len(m)} nodes: {int((m.HW_source == 'measured').sum())} measured, "
          f"{len(miss)} not\n")
    print("  the unmeasured nodes, by distance to the nearest crossing")
    b = pd.cut(miss.d_cross, [-1, 15, 30, 50, 1e6],
               labels=["<15 m", "15-30 m", "30-50 m", ">50 m"])
    t = b.value_counts().sort_index()
    for k, v in t.items():
        print(f"    {k:<10}{v:>5}   {v/len(miss)*100:>5.1f}%")

    print("\n  by street")
    s = m.assign(ok=m.HW_source == "measured").groupby("osm_name").ok.agg(
        ["sum", "count"])
    s.columns = ["measured", "nodes"]
    s["unmeasured"] = s.nodes - s.measured
    s["pct"] = (s.unmeasured / s.nodes * 100).round(0)
    print(s.sort_values("unmeasured", ascending=False).to_string())

    fig, ax = plt.subplots(figsize=(13, 11))
    bf.plot(ax=ax, color="#E4E9E5", edgecolor="#CFD8D2", linewidth=0.4, zorder=1)
    for src in ("measured", "segment_median", "none"):
        g = m[m.HW_source == src]
        if g.empty:
            continue
        g.plot(ax=ax, color=COL[src], markersize=26 if src != "measured" else 15,
               marker="o" if src == "measured" else "D",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_axis_off()
    ax.set_title("H/W coverage by node\n"
                 f"{int((m.HW_source=='measured').sum())} measured, "
                 f"{int((m.HW_source=='segment_median').sum())} inherited, "
                 f"{int((m.HW_source=='none').sum())} absent",
                 fontsize=13, loc="left", pad=14)
    ax.legend(handles=[Line2D([], [], marker="o" if k == "measured" else "D",
                              color="none", markerfacecolor=COL[k],
                              markeredgecolor="white", markersize=9, label=LAB[k])
                       for k in COL],
              loc="lower left", frameon=False, fontsize=10)
    out = RES / "figures" / "hw_coverage_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
