"""How the street width is measured, and the two probes that came before.

W is the denominator of H/W, and it is measured rather than assigned. An
earlier version set W by typology -- 30.5 m for avenues, 18.3 m for cross
streets -- which made the enclosure ratio partly definitional: typology fixed
the denominator, so a typology contrast in H/W was guaranteed. Measuring
facade to facade removes that circularity, and reverses the result.

Three probes were tried, and each failed in a way that motivated the next.

  LINE. A single zero-width ray perpendicular to the street. It slides between
  footprint polygons and past building corners: 12 per cent of mid-block nodes
  came back with no wall on one side.

  CONE. A wedge, ten degrees either side. The error being tolerated is angular
  -- a mis-estimated street bearing displaces a ray in proportion to distance
  -- so the tolerance is angular too, 1.2 m wide at 14 m and 7.9 m at 90 m.
  That fixes the sliding and creates a new failure: because it widens with
  range it can pass through a gap at a crossing and find a building on the
  next block, returning 110 m at n00377, on a street whose corner buildings
  stand 10 m away.

  BAND, reduced by the nearest hit. A parallel-sided corridor one node spacing
  wide -- 20 m -- sampled by nine rays. It does not widen, so it cannot reach
  the next block, and adjacent nodes tile the street rather than overlapping,
  so no frontage is measured twice.

WHY NOT THE CONE, GIVEN IT CORRELATES BETTER. Pooled over all nodes the cone
does: rho +0.543 against band+min's +0.482 versus measured enclosure. That
comparison is a Simpson's paradox. Split by whether a node sits at a crossing,
band+min matches the imagery at least as well in every subgroup, and every
within-group interval spans zero. The cone's pooled lead comes from returning
enormous widths at crossings, which splits the data into two clumps that track
crossing-ness rather than width. It is rewarded for encoding a category, not
for measuring a distance.

    .venv/Scripts/python tools/width_probe_diagram.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RAW, PROJ_CRS, banner, street_axis

BG, FG, MUT = "#ffffff", "#1a1a1a", "#6b7078"
BLD = "#d9d9de"
HIT = "#c0392b"
RAY = "#2f6fb0"


def cast(pt, bearing, reach, tree, geoms):
    """Nearest facade along one ray, or None."""
    ux, uy = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    ray = LineString([(pt.x, pt.y), (pt.x + ux * reach, pt.y + uy * reach)])
    best = None
    for k in tree.query(ray):
        g = geoms[k]
        if not ray.intersects(g):
            continue
        inter = ray.intersection(g)
        for p in (inter.geoms if hasattr(inter, "geoms") else [inter]):
            for c in np.asarray(p.coords) if hasattr(p, "coords") else []:
                d = float(np.hypot(c[0] - pt.x, c[1] - pt.y))
                if d > 0.5 and (best is None or d < best[0]):
                    best = (d, c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=None, help="node to draw; default picks "
                                                 "one where the probes disagree")
    ap.add_argument("--reach", type=float, default=90.0)
    ap.add_argument("--slide", default="16:9", choices=["16:9", "free"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/width_probe.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    banner("how the street width is measured")

    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(PROJ_CRS)
    fp = gpd.read_file(RAW / "building_footprints.geojson").to_crs(PROJ_CRS)
    geoms = list(fp.geometry.values)
    tree = STRtree(geoms)

    # n00641, East 42nd Street, MID-BLOCK. A crossing node makes the cone look
    # worse than it is -- its 110 m at n00377 is the extreme case and invites
    # the objection that crossings are special. Mid-block the same three
    # failures are all present and none of them depends on an intersection:
    # the line finds no wall on one side, the cone over-measures at 43.2 m,
    # and the band returns 30.9 m against the pipeline's own 30.7.
    nid = args.node or "n00641"
    if nid not in set(nodes.node_id):
        nid = nodes.node_id.iloc[len(nodes) // 2]
    row = nodes[nodes.node_id == nid].iloc[0]
    pt = Point(row.geometry.x, row.geometry.y)
    # Street bearing from the chain's own geometry. common.street_axis has a
    # different signature here and silently returned 0, which pointed every
    # probe due north instead of across the street -- and the figure still drew,
    # which is exactly the kind of wrong that looks right.
    chain = nodes[nodes.chain == row.chain].copy()
    cx = chain.geometry.x.to_numpy(); cy = chain.geometry.y.to_numpy()
    v = np.linalg.svd(np.c_[cx - cx.mean(), cy - cy.mean()],
                      full_matrices=False)[2][0]
    axis = float(np.degrees(np.arctan2(v[0], v[1])) % 180)
    perp = (axis + 90.0) % 360
    print(f"node {nid} on {row.get('osm_name', row.chain)}, "
          f"street axis {axis:.1f} deg")

    cone_deg = CFG["geometry"]["facade_cone_deg"]
    band_w = CFG["sampling"]["grid_spacing_m"]
    n_rays = CFG["geometry"].get("hw_band_rays", 9)

    size = (13.333, 7.5) if args.slide == "16:9" else (13.5, 5.2)
    fig, axes = plt.subplots(1, 3, figsize=size, facecolor=BG)

    # Short lines, hard-wrapped. Long single-line captions ran into each
    # other's panels, which no amount of figure width fixes.
    titles = [
        ("1  line",
         "one ray each way\nslips through the gap:\n12% of mid-block nodes\n"
         "found no wall on a side"),
        ("2  cone",
         f"+/-{cone_deg} deg, widening\nabsorbs a bearing error\n"
         "but over-reaches through\nthe same gap"),
        ("3  band + nearest",
         f"{band_w:.0f} m corridor, {n_rays} rays\nnever widens, so it cannot\n"
         "over-reach; adjacent nodes\ntile instead of overlapping"),
    ]

    for ax, (name, blurb), mode in zip(axes, titles, ("line", "cone", "band")):
        ax.set_facecolor(BG)
        win = 70
        ax.set_xlim(pt.x - win, pt.x + win)
        ax.set_ylim(pt.y - win, pt.y + win)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c8c8cc")

        sub = fp[fp.intersects(Point(pt).buffer(win * 1.6))]
        sub.plot(ax=ax, color=BLD, edgecolor="#b9b9c0", linewidth=.6, zorder=1)

        widths = []
        for sign in (0, 180):
            b = (perp + sign) % 360
            if mode == "line":
                hits = [cast(pt, b, args.reach, tree, geoms)]
                for dd in np.array([0.0]):
                    ux, uy = np.sin(np.radians(b)), np.cos(np.radians(b))
                    ax.plot([pt.x, pt.x + ux * args.reach],
                            [pt.y, pt.y + uy * args.reach],
                            color=RAY, lw=1.1, zorder=3)
            elif mode == "cone":
                offs = np.linspace(-cone_deg, cone_deg, 9)
                hits = [cast(pt, (b + o) % 360, args.reach, tree, geoms)
                        for o in offs]
                for o in offs:
                    ux, uy = np.sin(np.radians(b + o)), np.cos(np.radians(b + o))
                    ax.plot([pt.x, pt.x + ux * args.reach],
                            [pt.y, pt.y + uy * args.reach],
                            color=RAY, lw=.5, alpha=.55, zorder=3)
            else:
                # DISTANCE MEASURED FROM THE NODE, not from each ray's own
                # origin. s05 does the latter -- every ray is perpendicular to
                # the street, so its reach IS a street width at that offset,
                # and the minimum is the narrowest cross-section in the 20 m
                # the node stands for. This measures the shortest straight line
                # from the node to any facade the corridor touches, which is a
                # proximity rather than a width and is always the shorter of
                # the two. Drawn this way on request; the pipeline still
                # computes the perpendicular version.
                ux, uy = np.sin(np.radians(b)), np.cos(np.radians(b))
                px, py = np.sin(np.radians(axis)), np.cos(np.radians(axis))
                hits = []
                for off in np.linspace(-band_w / 2, band_w / 2, n_rays):
                    ox, oy = pt.x + px * off, pt.y + py * off
                    h = cast(Point(ox, oy), b, args.reach, tree, geoms)
                    if h is not None:
                        c = h[1]
                        hits.append((float(np.hypot(c[0] - pt.x, c[1] - pt.y)), c))
                    else:
                        hits.append(None)
                    ax.plot([ox, ox + ux * args.reach], [oy, oy + uy * args.reach],
                            color=RAY, lw=.5, alpha=.55, zorder=3)
            good = [h for h in hits if h]
            if good:
                d, c = min(good, key=lambda h: h[0])
                widths.append(d)
                # The nearest hit can lie on any ray of the corridor, and a dot
                # left out there reads as "measured from the far edge of the
                # band". The DISTANCE is what the probe returns, so it is drawn
                # back along the node's own perpendicular: same number, and it
                # reads as the width of the street at this node.
                if mode == "band":
                    # straight to the wall it found, which is the quantity now
                    ax.plot([pt.x, c[0]], [pt.y, c[1]], color=HIT, lw=2.6,
                            zorder=5, solid_capstyle="butt")
                    ax.plot([c[0]], [c[1]], "o", color=HIT, ms=6.5, zorder=6)
                else:
                    ux, uy = np.sin(np.radians(b)), np.cos(np.radians(b))
                    ax.plot([pt.x, pt.x + ux * d], [pt.y, pt.y + uy * d],
                            color=HIT, lw=2.6, zorder=5, solid_capstyle="butt")
                    ax.plot([pt.x + ux * d], [pt.y + uy * d], "o", color=HIT,
                            ms=6.5, zorder=6)
                    ax.plot([c[0]], [c[1]], "o", color=HIT, ms=3.5, alpha=.45,
                            zorder=5)
            else:
                widths.append(None)

        ax.plot([pt.x], [pt.y], "o", color="#111", ms=7, zorder=6)
        w = (sum(widths) if all(x is not None for x in widths) else None)
        got = f"W = {w:.1f} m" if w else "one side unresolved"
        ax.set_title(f"{name}\n{blurb}", color=FG, fontsize=9.5, pad=9,
                     linespacing=1.5)
        ax.text(.5, .02, got, transform=ax.transAxes, color=HIT if not w else FG,
                fontsize=12, ha="center", weight="bold")

    fig.text(.5, .975, f"measuring W at {nid}: three probes, same mid-block "
             f"node, same footprints", color=FG, fontsize=13.5, ha="center",
             va="top")
    fig.tight_layout(rect=[.01, .01, .99, .845])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
