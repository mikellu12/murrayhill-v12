"""The sampling nodes on a basemap, for either study area.

A locator: where the measurements were taken, and at what spacing. The nodes
are the study -- every rating and every segmentation share is attached to one
of these points -- so a reader who has seen this knows what the other figures
are made of.

Tiles are CARTO (OpenStreetMap data, ODbL), openly licensed and attributable,
unlike a screenshot of a commercial web map, which cannot go in a manuscript.
They cache under the study area's own processed/ folder, so repeat runs are
offline.

    .venv/Scripts/python tools/node_map.py
    SIM_CONFIG=config_london.yaml .venv/Scripts/python tools/node_map.py --dark
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner

WEB = 3857


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", default=True)
    ap.add_argument("--light", dest="dark", action="store_false")
    ap.add_argument("--colour", default="#f5c518")
    ap.add_argument("--size", type=float, default=13.0)
    ap.add_argument("--pad", type=float, default=0.16,
                    help="context around the frame, as a fraction of its span")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    default=True, help="refetch tiles rather than trusting "
                                       "the cache, which can hold error tiles")
    args = ap.parse_args()
    name = CFG.get("study_area_name", "study area")
    banner(f"sampling nodes: {name}")

    import contextily as cx
    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(WEB)
    x, y = nodes.geometry.x.values, nodes.geometry.y.values
    span = max(np.ptp(x), np.ptp(y))
    cxm, cym = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    h = span / 2 * (1 + args.pad * 2)
    print(f"{len(nodes)} nodes")

    fg = "#ffffff" if args.dark else "#111111"
    bg = "#0b0b0d" if args.dark else "#ffffff"
    src = (cx.providers.CartoDB.DarkMatterNoLabels if args.dark
           else cx.providers.CartoDB.PositronNoLabels)
    lbl = (cx.providers.CartoDB.DarkMatterOnlyLabels if args.dark
           else cx.providers.CartoDB.PositronOnlyLabels)

    fig, ax = plt.subplots(figsize=(9.6, 9.6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.set_xlim(cxm - h, cxm + h); ax.set_ylim(cym - h, cym + h)
    ax.set_aspect("equal"); ax.set_axis_off()

    # THE CACHE CAN POISON THE FIGURE. contextily stores whatever came back,
    # including a provider's "requires API key" placeholder, and then serves it
    # forever without another request -- the London map carried that watermark
    # across the whole basemap while every live tile fetched fine. --no-cache
    # bypasses it; deleting the directory is the cure once it has happened.
    cache = PROC / "tiles"
    if args.cache:
        cache.mkdir(parents=True, exist_ok=True)
        cx.set_cache_dir(str(cache))
    for prov, alpha in ((src, 1.0), (lbl, 0.85)):
        try:
            cx.add_basemap(ax, source=prov, alpha=alpha, attribution=False)
        except Exception as e:
            print(f"  basemap layer skipped: {e}")

    # A dark basemap that comes back bright is a page of placeholders, not a
    # map. Cheap to check, and it fails silently otherwise.
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float)
    if args.dark and buf.mean() > 110:
        print(f"  WARNING: basemap mean brightness {buf.mean():.0f} on a dark "
              f"style -- these are probably error tiles. Re-run with "
              f"--no-cache, or delete {cache}")

    ax.scatter(x, y, s=args.size, c=args.colour, linewidths=0, zorder=5)

    # SPACING IS MEASURED WITHIN A STREET, not globally. The global
    # nearest-neighbour distance in the City of London is 12.0 m, which reads
    # as over-sampling against the manuscript's 20 m and is not: it is nodes on
    # DIFFERENT streets sitting close together, which is what the City is --
    # Abchurch Lane and King William Street are metres apart. Along a street
    # the spacing is 18.4 m. Printing the global figure on the map would put a
    # wrong number under the reader's eye.
    m = nodes.to_crs(CFG.get("projected_crs", 32618))
    from scipy.spatial import cKDTree
    xy = np.c_[m.geometry.x, m.geometry.y]
    key = ("street_name" if "street_name" in nodes.columns
           else "osm_name" if "osm_name" in nodes.columns else None)
    d, i = cKDTree(xy).query(xy, k=min(12, len(xy)))
    if key:
        lab = nodes[key].astype(str).values
        same = []
        for r in range(len(xy)):
            for k in range(1, d.shape[1]):
                if lab[i[r, k]] == lab[r]:
                    same.append(d[r, k]); break
        step = float(np.median(same))
        print(f"spacing along a street: median {step:.1f} m "
              f"(global nearest neighbour {np.median(d[:, 1]):.1f} m, which is "
              f"adjacent streets, not sampling)")
    else:
        step = float(np.median(d[:, 1]))
        print(f"nearest-neighbour spacing: median {step:.1f} m")

    ax.text(.03, .965, name, transform=ax.transAxes, color=fg, fontsize=19,
            va="top")
    ax.text(.03, .925, f"{len(nodes)} sampling nodes, {step:.0f} m apart "
            f"along each street",
            transform=ax.transAxes, color=fg, fontsize=11, va="top", alpha=.78)
    ax.text(.985, .012, "basemap: CARTO, OpenStreetMap contributors (ODbL)",
            transform=ax.transAxes, color=fg, fontsize=7, ha="right", alpha=.55)

    out = args.out or (RES / "figures" /
                       f"node_map_{CFG.get('study_area_slug', 'area')}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=bg, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
