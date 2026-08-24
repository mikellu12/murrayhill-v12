"""Murray Hill in black and white, with the street space drawn as a figure.

The figure-ground plan shows built mass against void, which is the right
drawing for a massing argument but leaves the street reading as background:
a courtyard, a parking lot and a carriageway are all just white. This one
demarcates the street explicitly -- centrelines are buffered to their real
width and outlined, so the public right of way is a bounded shape rather
than whatever is left over between buildings.

Manhattan's widths come from the 1811 grid and are close to uniform: the
avenues were laid out at 100 ft and the numbered cross streets at 60 ft, so
half-widths of 15 m and 9 m reproduce the corridor without needing a
per-segment width field that CSCL does not carry.

No colour anywhere, and no grey doing work that a line should do: black
mass, white void, hairline kerbs.

    .venv/Scripts/python tools/site_map_bw.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RAW, RES, banner

GIS = PROC / "nyc_gis"
AVENUE_HALF_M = 15.0        # 100 ft right of way
STREET_HALF_M = 9.0         # 60 ft


def _corridors() -> gpd.GeoDataFrame:
    """Street centrelines buffered to their right of way, dissolved."""
    cl = gpd.read_file(GIS / "centerline_mh.gpkg").to_crs(32618)
    name = cl.get("stname_label", cl.get("street", "")).astype(str).str.upper()
    half = np.where(name.str.contains("AVENUE|AVE\\b|BROADWAY", regex=True),
                    AVENUE_HALF_M, STREET_HALF_M)
    cl["geometry"] = [g.buffer(h, cap_style=2, join_style=2)
                      for g, h in zip(cl.geometry, half)]
    return gpd.GeoDataFrame(geometry=[cl.union_all()], crs=32618).to_crs(4326)


def _sampled_lines(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for (name, chain), g in nodes.groupby(["osm_name", "chain"]):
        g = g.sort_values("chain_pos_m")
        if len(g) > 1:
            rows.append({"geometry": LineString(list(zip(g.geometry.x, g.geometry.y)))})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes.crs)


def draw(nodes: gpd.GeoDataFrame, filled: bool) -> None:
    """One drawing. filled=True gives solid mass; False gives line work only.

    Unfilled is the survey convention: the block is described by its edges
    and the page stays open, which suits a figure that has to carry labels
    and sample points on top. Filled is the massing convention, where the
    built fabric is the subject and everything else recedes. They answer
    different questions, so both are produced.
    """
    ll = nodes.to_crs(4326)

    fig = plt.figure(figsize=(9.0, 8.4), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    # Street space first, as a bounded white figure with a kerb line. When
    # nothing is filled the kerb carries the whole demarcation, so it is
    # drawn heavier than the building edges rather than matching them.
    _corridors().plot(ax=ax, facecolor="white", edgecolor="black",
                      linewidth=0.45 if filled else 0.75, zorder=2)

    # Built mass on top: where a building overlaps the buffered corridor the
    # building wins, which is what keeps the kerb line reading as the
    # building line rather than as an offset from the centre of the road.
    footprints = RAW / "building_footprints.geojson"
    if footprints.exists():
        gpd.read_file(footprints).to_crs(4326).plot(
            ax=ax,
            facecolor="black" if filled else "none",
            edgecolor="black",
            linewidth=0.2 if filled else 0.35, zorder=3)

    _sampled_lines(nodes).to_crs(4326).plot(ax=ax, color="black", linewidth=0.9,
                                            linestyle=(0, (6, 3)), zorder=4)
    ax.scatter(ll.geometry.x, ll.geometry.y, s=7.0, facecolor="black",
               edgecolor="white", linewidths=0.55, zorder=5)

    for name, g in ll.groupby("osm_name"):
        avenue = ("Avenue" in name
                  or name in ("Tudor City Place", "Tunnel Exit Street"))
        i = g.geometry.y.idxmax() if avenue else g.geometry.x.idxmin()
        pt = g.loc[i].geometry
        label = (name.replace(" Street", "").replace("East ", "E ")
                     .replace(" Avenue", " Ave")
                     .replace("Tudor City Place", "Tudor City"))
        ax.annotate(label, (pt.x, pt.y), fontsize=7, color="black",
                    ha="center" if avenue else "right",
                    va="bottom" if avenue else "center",
                    xytext=(0, 8) if avenue else (-7, 0),
                    textcoords="offset points",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white",
                              ec="black", lw=0.35))

    mx = 0.0018
    ax.set_xlim(ll.geometry.x.min() - mx * 1.7, ll.geometry.x.max() + mx * 1.7)
    ax.set_ylim(ll.geometry.y.min() - mx, ll.geometry.y.max() + mx)

    m200 = 200 / (111_320 * np.cos(np.radians(40.75)))
    x0 = ll.geometry.x.min() - mx * 1.4
    y0 = ll.geometry.y.min() - mx * 0.55
    ax.plot([x0, x0 + m200], [y0, y0], color="black", lw=1.8, zorder=6)
    ax.annotate("200 m", (x0 + m200 / 2, y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7.5, color="black")
    ax.annotate("N", (ll.geometry.x.max() + mx * 1.2, ll.geometry.y.max() - mx * .2),
                fontsize=9, color="black", ha="center", fontweight="bold", zorder=6)
    ax.annotate("", (ll.geometry.x.max() + mx * 1.2, ll.geometry.y.max() - mx * .55),
                xytext=(ll.geometry.x.max() + mx * 1.2, ll.geometry.y.max() - mx * 1.9),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=6)

    ax.set_aspect(1 / np.cos(np.radians(40.75)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Murray Hill  —  {len(nodes)} sampling nodes at 20 m",
                 fontsize=11.5, color="black", loc="left", pad=10)

    stem = "figure_site_murrayhill_bw" + ("" if filled else "_outline")
    for ext in ("png", "pdf"):
        out = RES / "figures" / f"{stem}.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    banner("Murray Hill, black and white")
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in nodes.columns:
        nodes = nodes[nodes.in_study]
    for filled in (True, False):
        draw(nodes, filled)
    print(f"\n{len(nodes)} nodes, {nodes.osm_name.nunique()} streets")


if __name__ == "__main__":
    main()
