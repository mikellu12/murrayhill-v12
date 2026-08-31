"""Murray Hill on a desaturated basemap, in the reference style.

A locator drawn the way planning agencies draw one: the whole city rendered
flat and grey so it reads as context rather than content, with the study
area outlined in a single saturated colour on top. Everything needed to
place the site -- the rest of the street network, the shoreline,
neighbourhood names -- is present but visually subordinate, so the eye goes
to the outline first and the context answers "where" only once asked.

That is the opposite of the figure-ground and line-work drawings, which
strip context away to make block structure the subject. Both are correct
for different jobs: this one situates, those describe.

Tiles are CARTO Positron (OpenStreetMap data, ODbL), openly licensed and
attributed on the figure -- unlike a screenshot of a commercial web map,
which cannot be published in a manuscript. Tiles cache under
data/processed/tiles, so repeat runs are offline and re-request nothing.

    .venv/Scripts/python tools/site_map_basemap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import LineString, box

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, CFG, PROC, RES, banner

GIS = PROC / "nyc_gis"
TILES = PROC / "tiles"
WEB = 3857

BLUE = "#1f8fe8"          # the boundary colour in the reference figure
DEEP = "#0d5c99"
INK = "#1b1f24"

# Black and white is not the colour figure with the saturation pulled out.
# The blue was carrying the whole figure/context split on its own; without
# hue, weight has to carry it, so the mono outlines are drawn heavier and
# the area wash is dropped to a hairline-bounded tint that survives a
# greyscale press.
# The borough outline has to read as the island's edge at a glance without
# competing with the study box. A hairline disappeared into the basemap's own
# linework; the earlier heavy weight fought the box. This sits between, still
# lighter than the box so the hierarchy holds.
MONO = dict(edge="#000000", deep="#000000", wash="#000000",
            wash_alpha=0.10, lw_boro=1.35, lw_study=2.0)
COLOUR = dict(edge=BLUE, deep=DEEP, wash=BLUE,
              wash_alpha=0.20, lw_boro=1.35, lw_study=1.9)

# Positron carries labels; NoLabels is the same cartography without them.
# Which is right depends on the frame: at city scale the place names are the
# whole point of a locator, and at block scale they collide with the
# figure's own labels at a different size and colour.
PROVIDER = cx.providers.CartoDB.Positron
PROVIDER_CLEAN = cx.providers.CartoDB.PositronNoLabels


def _study_box(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The study area as a rectangle turned onto the Manhattan grid.

    A north-aligned box is the wrong shape for this city. The 1811 grid runs
    about 29 degrees off true north, so an axis-aligned rectangle drawn
    around a sample that follows the grid has to be oversized to contain it
    and still reads as tilted against every street inside it. The minimum
    rotated rectangle of the sampled nodes takes its angle from the sample
    itself, so its sides run parallel to the avenues and the cross streets.

    Fitted to the nodes rather than to config's bbox for the same reason the
    detail map uses the node hull: the bbox is a fetch window, larger than
    what was actually measured.
    """
    g = nodes.to_crs(PROJ_CRS)
    rect = g.buffer(35).union_all().minimum_rotated_rectangle
    return gpd.GeoDataFrame(geometry=[rect], crs=PROJ_CRS).to_crs(WEB)


def _hull(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The sampled extent: nodes buffered and dissolved.

    The bbox is a fetch window and is larger than what was measured, so for
    the close view the honest outline is the sample's own footprint rather
    than the download request.
    """
    g = nodes.to_crs(PROJ_CRS)
    return gpd.GeoDataFrame(geometry=[g.buffer(45).union_all().buffer(-15)],
                            crs=PROJ_CRS).to_crs(WEB)


def _sampled_lines(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for _, g in nodes.groupby(["osm_name", "chain"]):
        g = g.sort_values("chain_pos_m")
        if len(g) > 1:
            rows.append({"geometry": LineString(list(zip(g.geometry.x, g.geometry.y)))})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes.crs).to_crs(WEB)


def _basemap(ax, provider, zoom, mono=False, black=185):
    """Draw the tile basemap, optionally stripped to luminance.

    Positron is nearly grey already, but "nearly" is what shows up on a
    journal's greyscale press as a muddy tint: its parks are green and its
    water blue. Converting to luminance makes the drawing genuinely
    achromatic instead of hoping the colours are pale enough not to matter.

    Luminance is Rec. 709 weighted, not a channel average -- an average
    lightens blue water and darkens green parks relative to how the eye
    reads them, which inverts their apparent order. The result is then
    stretched, because Positron's whole range sits in the top fifth of the
    scale and collapses to a flat field once the hue is gone.

    How hard to stretch depends on what fills the frame, so `black` is set
    per figure rather than fixed. The city view spans water to paper and
    wants the harder stretch; the block view is almost entirely built fabric in a
    narrow band near white, and pulling it as hard turns every building
    mid-grey and swallows the sample drawn on top.
    """
    if not mono:
        cx.add_basemap(ax, source=provider, zoom=zoom, attribution=False,
                       zorder=0, interpolation="bilinear")
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    img, ext = cx.bounds2img(xlim[0], ylim[0], xlim[1], ylim[1],
                             zoom=zoom, source=provider, ll=False)
    rgb = img[..., :3].astype(float)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ax.imshow(lum, extent=ext, cmap="gray", vmin=black, vmax=252,
              zorder=0, interpolation="bilinear")
    ax.set_xlim(*xlim)                    # imshow resets the frame
    ax.set_ylim(*ylim)


def _scalebar(ax, metres, label):
    """A bar in projected units, on a plate.

    Web Mercator inflates ground distance by 1/cos(lat), so a bar drawn at
    the raw metre value would be short by a third at this latitude.

    Set on an opaque plate and inboard of the frame edge: a reader looking
    at this on a phone sees a centre crop, and furniture pinned to the
    extreme corner is the first thing lost.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = metres / np.cos(np.radians(40.75))
    bx = x0 + (x1 - x0) * 0.06
    by = y0 + (y1 - y0) * 0.075
    h = (y1 - y0) * 0.006
    # A checked bar reads as a scale at a glance where a plain rule reads as
    # a stray line; halves are enough at this size.
    ax.add_patch(Rectangle((bx - (x1 - x0) * 0.022, by - (y1 - y0) * 0.028),
                           length + (x1 - x0) * 0.044, (y1 - y0) * 0.072,
                           facecolor="white", edgecolor="none",
                           alpha=0.85, zorder=8))
    for k in (0, 1):
        ax.add_patch(Rectangle((bx + k * length / 2, by), length / 2, h,
                               facecolor=INK if k == 0 else "white",
                               edgecolor=INK, linewidth=0.8, zorder=9))
    for frac, txt in ((0.0, "0"), (1.0, label)):
        ax.annotate(txt, (bx + frac * length, by + h), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK, zorder=9)


def _north(ax):
    """North arrow, on a plate and inboard for the same reason."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    nx = x1 - (x1 - x0) * 0.075
    ny = y1 - (y1 - y0) * 0.145
    span = (y1 - y0) * 0.075
    ax.add_patch(Rectangle((nx - (x1 - x0) * 0.030, ny - (y1 - y0) * 0.022),
                           (x1 - x0) * 0.060, span + (y1 - y0) * 0.075,
                           facecolor="white", edgecolor="none",
                           alpha=0.85, zorder=8))
    ax.annotate("", (nx, ny + span), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                mutation_scale=16), zorder=9)
    ax.annotate("N", (nx, ny + span), xytext=(0, 5),
                textcoords="offset points", fontsize=11, color=INK,
                ha="center", fontweight="bold", zorder=9)


def _credit(ax, extra="", mono=False):
    """Attribution, set on a plate so it survives whatever it lands on.

    In the mono figures the bottom right corner is open water, which the
    contrast stretch renders mid-grey; grey type on it is unreadable.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate(f"Basemap CARTO / OpenStreetMap contributors{extra}",
                (x1 - (x1 - x0) * 0.012, y0 + (y1 - y0) * 0.013),
                fontsize=5.6, color="#5a5a5a" if mono else "#8b939c",
                ha="right", zorder=9,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="none", alpha=0.82 if mono else 0.0))


def _finish(ax, title):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d2d7dd")
        sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=8)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        out = RES / "figures" / f"{stem}.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def context(nodes, mono=False):
    """Manhattan and its neighbours, study area outlined."""
    c = MONO if mono else COLOUR
    fig = plt.figure(figsize=(9.4, 8.6), facecolor="white")
    ax = fig.add_subplot(111)

    # Framed so enough of the other boroughs show that the island reads as
    # part of a city rather than as a shape floating on grey.
    b = CFG["study_area"]["bbox"]
    mid_x, mid_y = (b["west"] + b["east"]) / 2, (b["south"] + b["north"]) / 2
    focus = gpd.GeoDataFrame(
        geometry=[box(mid_x - .175, mid_y - .135, mid_x + .175, mid_y + .135)],
        crs=4326).to_crs(WEB)
    xmin, ymin, xmax, ymax = focus.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _basemap(ax, PROVIDER, zoom=13, mono=mono)

    boro = gpd.read_file(GIS / "boroughs.gpkg").to_crs(WEB)
    boro[boro.boroname == "Manhattan"].boundary.plot(
        ax=ax, color=c["edge"], linewidth=c["lw_boro"], zorder=3)

    study = _study_box(nodes)
    study.plot(ax=ax, facecolor=c["wash"], alpha=c["wash_alpha"],
               edgecolor="none", zorder=4)
    study.boundary.plot(ax=ax, color=c["deep"], linewidth=c["lw_study"], zorder=5)
    # No sampled streets drawn here. At 2 km the seventeen of them cover
    # about four millimetres and resolve to a smudge rather than to streets,
    # which reads as a printing artefact inside the box. The street network
    # belongs on the detail map, where it is legible.

    p = study.geometry.iloc[0]
    ax.annotate("Murray Hill", (p.bounds[2], (p.bounds[1] + p.bounds[3]) / 2),
                xytext=(48, 0), textcoords="offset points", va="center",
                fontsize=10.5, color=c["deep"], fontweight="semibold", zorder=8,
                arrowprops=dict(arrowstyle="-", color=c["deep"], lw=1.3),
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec=c["deep"], lw=0.7, alpha=.95))

    _scalebar(ax, 2000, "2 km")
    _north(ax)
    _credit(ax, ".  Boundary: NYC DCP", mono)
    _finish(ax, "Murray Hill in Manhattan")
    _save(fig, "figure_site_basemap_context" + ("_bw" if mono else ""))


def detail(nodes, mono=False):
    """The study area itself, over the same grey cartography."""
    c = MONO if mono else COLOUR
    ll = nodes.to_crs(WEB)
    fig = plt.figure(figsize=(9.0, 8.4), facecolor="white")
    ax = fig.add_subplot(111)

    mx = (ll.geometry.x.max() - ll.geometry.x.min()) * 0.10
    ax.set_xlim(ll.geometry.x.min() - mx * 1.5, ll.geometry.x.max() + mx * 1.5)
    ax.set_ylim(ll.geometry.y.min() - mx * .8, ll.geometry.y.max() + mx * .8)
    _basemap(ax, PROVIDER_CLEAN, zoom=17, mono=mono, black=140)

    hull = _hull(nodes)
    hull.plot(ax=ax, facecolor=c["wash"], alpha=c["wash_alpha"] * 0.65,
              edgecolor="none", zorder=2)
    hull.boundary.plot(ax=ax, color=c["edge"], linewidth=c["lw_study"], zorder=3)

    _sampled_lines(nodes).plot(ax=ax, color=c["deep"], linewidth=2.0, zorder=4)
    ax.scatter(ll.geometry.x, ll.geometry.y, s=9, facecolor=c["deep"],
               edgecolor="white", linewidths=.6, zorder=5)

    for name, g in ll.groupby("osm_name"):
        avenue = ("Avenue" in name
                  or name in ("Tudor City Place", "Tunnel Exit Street"))
        i = g.geometry.y.idxmax() if avenue else g.geometry.x.idxmin()
        pt = g.loc[i].geometry
        label = (name.replace(" Street", "").replace("East ", "E ")
                     .replace(" Avenue", " Ave")
                     .replace("Tudor City Place", "Tudor City"))
        ax.annotate(label, (pt.x, pt.y), fontsize=7, color=INK,
                    ha="center" if avenue else "right",
                    va="bottom" if avenue else "center",
                    xytext=(0, 15) if avenue else (-10, 0),
                    textcoords="offset points", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#c8ced5", lw=0.4, alpha=.92))

    _scalebar(ax, 200, "200 m")
    _north(ax)
    _credit(ax, mono=mono)
    _finish(ax, f"Murray Hill  —  {len(nodes)} sampling nodes at 20 m")
    _save(fig, "figure_site_basemap_detail" + ("_bw" if mono else ""))


def main():
    banner("basemap site maps")
    TILES.mkdir(parents=True, exist_ok=True)
    cx.set_cache_dir(str(TILES))
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in nodes.columns:
        nodes = nodes[nodes.in_study]
    for mono in (False, True):
        context(nodes, mono)
        detail(nodes, mono)
    print(f"\n{len(nodes)} nodes, {nodes.osm_name.nunique()} streets")
    print(f"tiles cached in {TILES}")


if __name__ == "__main__":
    main()
