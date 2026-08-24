"""Sampling-frame audit map: what is in the analytic sample and what is not.

Every absence in this frame has a reason, and the point of the map is that
they are visibly different kinds of absence: a capture-date exclusion is
scattered, a coverage failure would be a contiguous run, and a tunnel
segment is a deliberate flag rather than a gap at all.

Regenerating this after any frame change is not optional. It was briefly
possible to read a v12 node map captioned as v13, which is the kind of error
that survives review precisely because the picture looks plausible.

    .venv/Scripts/python tools/frame_audit.py
"""
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
from common import PROC, RES, banner


def main():
    banner("frame audit map")
    n = gpd.read_file(PROC / "nodes.gpkg")
    m = pd.read_csv(PROC / "metrics.csv")
    n["has"] = n.node_id.isin(m.node_id)
    if "in_study" in n.columns:
        outside = int((~n.in_study).sum())
        n = n[n.in_study].copy()
        print(f"study-area filter: dropped {outside} nodes outside the area")
    n["x"], n["y"] = n.geometry.x, n.geometry.y
    print(f"frame {len(n)}  analytic {int(n.has.sum())}  "
          f"tunnel-flagged {int(n.is_tunnel.sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    ok = n[n.has & ~n.is_tunnel]
    tun = n[n.has & n.is_tunnel]
    bad = n[~n.has]
    ax.scatter(ok.x, ok.y, s=11, c="#2e7d32", zorder=2,
               label=f"analytic sample ({len(ok)})")
    ax.scatter(tun.x, tun.y, s=13, c="#1565c0", zorder=3,
               label=f"tunnel-flagged, kept ({len(tun)})")
    ax.scatter(bad.x, bad.y, s=52, c="#d32f2f", marker="X", zorder=4,
               edgecolor="k", linewidth=.3,
               label=f"excluded ({len(bad)})")
    for st, g in n.groupby("osm_name"):
        i = g.y.idxmax()
        ax.annotate(st.replace(" Street", "").replace(" Avenue", " Ave"),
                    (g.loc[i, "x"], g.loc[i, "y"]), fontsize=6.5, color="#555",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8.5, loc="lower right", framealpha=.92)
    ax.set_title(f"v13 frame — {len(n)} nodes, {int(n.has.sum())} analytic",
                 fontsize=12, loc="left")

    axb = axes[1]
    d = n.merge(m[["node_id", "GVI"]], on="node_id")
    sc = axb.scatter(d.x, d.y, c=d.GVI, s=17, cmap="YlGn",
                     vmin=0, vmax=np.nanpercentile(d.GVI, 97))
    plt.colorbar(sc, ax=axb, label="GVI (%)", shrink=.75)
    axb.set_aspect("equal"); axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("GVI — Park Ave's median is the bright line",
                  fontsize=12, loc="left")

    fig.suptitle("Murray Hill v13 — sampling frame audit (MIKE_PC)",
                 fontsize=13, y=.97)
    out = RES / "figures" / "frame_audit.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    print("\n=== excluded, by street ===")
    print(bad.osm_name.value_counts().to_string() if len(bad) else "  none")


if __name__ == "__main__":
    main()
