"""Location figures for the paper: a city locator and a Murray Hill detail.

Two separate files, not two panels. A locator and a detail map answer
different questions and a manuscript rarely wants them at the same size or in
the same place; kept apart, each can be placed and scaled alone.

Geometry is New York City's own published GIS -- borough boundaries, parks
and the street centreline (CSCL) from NYC Open Data -- rather than a
screenshot of a web map. Three reasons that matters for a manuscript: the
data is openly licensed where a vendor basemap under a screenshot is not, it
renders at whatever resolution the journal asks for instead of the ~150 dpi
a screen capture gives, and it carries no interface furniture.

Sampled streets are drawn from the sampling frame rather than from CSCL, so
the map cannot disagree with the sample it illustrates: a street is drawn
bold if and only if it was measured, along exactly the stretch that was.
CSCL supplies the surrounding streets that were not.

Layers are cached under data/processed/nyc_gis, so this redraws offline.

    .venv/Scripts/python tools/site_maps.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import LineString

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RAW, RES, banner

INK, MID, LIGHT, PAPER = "#1b1f24", "#6b7480", "#c9cfd6", "#ffffff"
ACCENT = "#1f5c4c"
FILL, LAND, WATER, PARK = "#e6ece8", "#eceef0", "#dbe7ef", "#d8e6d9"
GIS = PROC / "nyc_gis"

# Placed by hand: a centroid puts a borough label in the wrong lobe, and the
# river labels belong in water where no polygon exists to anchor them.
BOROUGH_LABELS = [
    ("Manhattan", -73.949, 40.820),
    ("The Bronx", -73.868, 40.858),
    ("Queens", -73.812, 40.716),
    ("Brooklyn", -73.949, 40.640),
    ("New Jersey", -74.090, 40.740),
]
WATER_LABELS = [
    ("Hudson River", -74.022, 40.760, 62),
    ("East River", -73.928, 40.782, -52),
]


def _gis(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(GIS / f"{name}.gpkg").to_crs(4326)


def _street_lines(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One line per sampled run, drawn through the nodes themselves."""
    rows = []
    for (name, chain), g in nodes.groupby(["osm_name", "chain"]):
        g = g.sort_values("chain_pos_m")
        if len(g) < 2:
            continue
        rows.append({"osm_name": name, "chain": chain,
                     "geometry": LineString(list(zip(g.geometry.x, g.geometry.y)))})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes.crs)


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = RES / "figures" / f"{stem}.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", facecolor=PAPER)
        print(f"wrote {out}")
    plt.close(fig)


def _locator(bbox) -> None:
    fig = plt.figure(figsize=(6.2, 6.6), facecolor=PAPER)
    ax = fig.add_subplot(111)
    ax.set_facecolor(WATER)                      # land is drawn over water

    boroughs = _gis("boroughs")
    other = boroughs[boroughs.boroname != "Manhattan"]
    manhattan = boroughs[boroughs.boroname == "Manhattan"]
    other.plot(ax=ax, facecolor=LAND, edgecolor="#bcc4cb", linewidth=.5, zorder=1)
    manhattan.plot(ax=ax, facecolor=FILL, edgecolor=MID, linewidth=1.0, zorder=2)
    _gis("parks_mn").plot(ax=ax, facecolor=PARK, edgecolor="none", zorder=3)

    ax.add_patch(Rectangle((bbox["west"], bbox["south"]),
                           bbox["east"] - bbox["west"],
                           bbox["north"] - bbox["south"],
                           facecolor="none", edgecolor=ACCENT,
                           linewidth=2.0, zorder=6))
    ax.annotate("Murray Hill",
                (bbox["east"], (bbox["north"] + bbox["south"]) / 2),
                xytext=(34, 2), textcoords="offset points", va="center",
                fontsize=10, color=ACCENT, fontweight="semibold", zorder=7,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec="none", alpha=.92))

    for label, x, y in BOROUGH_LABELS:
        ax.annotate(label, (x, y), fontsize=8, color=MID, ha="center", zorder=5)
    for label, x, y, rot in WATER_LABELS:
        ax.annotate(label, (x, y), fontsize=7, color="#7d96ad", ha="center",
                    style="italic", rotation=rot, rotation_mode="anchor", zorder=5)

    # A north arrow and a scale bar are what separate a map from a shape.
    ax.annotate("N", (-73.793, 40.873), fontsize=9, color=INK,
                ha="center", fontweight="bold", zorder=7)
    ax.annotate("", (-73.793, 40.867), xytext=(-73.793, 40.838),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3), zorder=7)
    km5 = 5000 / (111_320 * np.cos(np.radians(40.75)))
    ax.plot([-74.068, -74.068 + km5], [40.582, 40.582], color=INK, lw=1.8, zorder=7)
    ax.annotate("5 km", (-74.068 + km5 / 2, 40.585), fontsize=7.5,
                color=INK, ha="center", zorder=7)
    ax.annotate("NYC Department of City Planning", (-73.768, 40.567),
                fontsize=5.8, color="#9aa3ac", ha="right", zorder=7)

    ax.set_xlim(-74.12, -73.76)
    ax.set_ylim(40.56, 40.90)
    ax.set_aspect(1 / np.cos(np.radians(40.75)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LIGHT)
    ax.set_title("New York City", fontsize=11.5, color=INK, loc="left", pad=8)
    _save(fig, "figure_site_locator")


# Three renderings of the same drawing. "grey" is a neutral GIS plan for a
# journal that prints in mono; "figureground" is the Nolli convention -- built
# mass solid, open space void -- which is how a morphology argument is
# normally drawn and makes the block structure the subject rather than the
# background.
STYLES = {
    "colour":       dict(bldg="#eef1f4", bldg_edge="#dfe4ea", other="#cdd3da",
                         street=MID, node=ACCENT, text=INK, bg=PAPER),
    "grey":         dict(bldg="#dcdcdc", bldg_edge="#bcbcbc", other="#c4c4c4",
                         street="#111111", node="#111111", text="#111111", bg=PAPER),
    "figureground": dict(bldg="#1a1a1a", bldg_edge="#1a1a1a", other="#9a9a9a",
                         street="#000000", node="#000000", text="#000000", bg=PAPER),
}


def _detail(nodes: gpd.GeoDataFrame, style: str = "colour") -> None:
    c = STYLES[style]
    nodes_ll = nodes.to_crs(4326)
    fig = plt.figure(figsize=(8.6, 8.0), facecolor=PAPER)
    ax = fig.add_subplot(111)
    ax.set_facecolor(c["bg"])

    footprints = RAW / "building_footprints.geojson"
    if footprints.exists():
        gpd.read_file(footprints).to_crs(4326).plot(
            ax=ax, facecolor=c["bldg"], edgecolor=c["bldg_edge"],
            linewidth=.25, zorder=1)

    # Every street in the area from CSCL, so the sampled ones read as a
    # selection out of a network rather than as the whole network.
    _gis("centerline_mh").plot(ax=ax, color=c["other"], linewidth=.7, zorder=2)
    _street_lines(nodes).to_crs(4326).plot(ax=ax, color=c["street"], linewidth=1.9,
                                           zorder=3, capstyle="round")
    # On a dark figure-ground the dots need a light rim to stay visible
    # where a street runs against built mass.
    ax.scatter(nodes_ll.geometry.x, nodes_ll.geometry.y, s=6.5 if style == "figureground" else 5.0,
               color=c["node"], zorder=4,
               edgecolors=PAPER if style == "figureground" else "none",
               linewidths=.5 if style == "figureground" else 0)

    # Cross-streets labelled at their west end, avenues at their north end:
    # where each run leaves the frame is where a label has room.
    for name, g in nodes_ll.groupby("osm_name"):
        avenue = ("Avenue" in name
                  or name in ("Tudor City Place", "Tunnel Exit Street"))
        i = g.geometry.y.idxmax() if avenue else g.geometry.x.idxmin()
        pt = g.loc[i].geometry
        label = (name.replace(" Street", "").replace("East ", "E ")
                     .replace(" Avenue", " Ave")
                     .replace("Tudor City Place", "Tudor City"))
        ax.annotate(label, (pt.x, pt.y), fontsize=7, color=c["text"],
                    ha="center" if avenue else "right",
                    va="bottom" if avenue else "center",
                    xytext=(0, 7) if avenue else (-6, 0),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc=PAPER, ec="none", alpha=.85))

    # Frame the sample, not the configured bbox: the bbox is a fetch window
    # and leaves a band of empty page above and below the grid.
    mx = 0.0016
    ax.set_xlim(nodes_ll.geometry.x.min() - mx * 1.6,
                nodes_ll.geometry.x.max() + mx * 1.6)
    ax.set_ylim(nodes_ll.geometry.y.min() - mx,
                nodes_ll.geometry.y.max() + mx)

    m200 = 200 / (111_320 * np.cos(np.radians(40.75)))
    x0 = nodes_ll.geometry.x.min() - mx * 1.3
    y0 = nodes_ll.geometry.y.min() - mx * .55
    ax.plot([x0, x0 + m200], [y0, y0], color=c["text"], lw=1.6, zorder=6)
    ax.annotate("200 m", (x0 + m200 / 2, y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7.5, color=c["text"])
    ax.annotate("Street centreline: NYC DCP (CSCL).  Footprints: NYC DOB.",
                (nodes_ll.geometry.x.max() + mx * 1.5,
                 nodes_ll.geometry.y.min() - mx * .85),
                fontsize=5.8, color="#9aa3ac", ha="right", zorder=6)

    ax.set_aspect(1 / np.cos(np.radians(40.75)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LIGHT)
    ax.set_title(f"Murray Hill  —  {len(nodes)} sampling nodes at 20 m",
                 fontsize=11.5, color=c["text"], loc="left", pad=8)
    suffix = "" if style == "colour" else f"_{style}"
    _save(fig, f"figure_site_murrayhill{suffix}")


def main() -> None:
    banner("site maps")
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in nodes.columns:
        nodes = nodes[nodes.in_study]
    _locator(CFG["study_area"]["bbox"])
    for style in STYLES:
        _detail(nodes, style)
    print(f"\n{len(nodes)} nodes, {nodes.osm_name.nunique()} streets")


if __name__ == "__main__":
    main()
