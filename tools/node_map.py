"""The sampling nodes on a basemap, for either study area.

A locator: where the measurements were taken, and at what spacing. The nodes
are the study -- every rating and every segmentation share is attached to one
of these points -- so a reader who has seen this knows what the other figures
are made of.

THE BASEMAP IS DRAWN FROM OSM GEOMETRY, not fetched as tiles. CARTO's tiles now
carry an "API KEY REQUIRED" watermark diagonally across every few tiles, and a
watermark cannot go in a manuscript. Switching to another tile host only moves
the dependency: any of them can start requiring a key, and the failure is
silent -- the figure renders, looks plausible, and is unusable.

Drawing the streets and water from OSM ourselves removes the dependency
entirely. It is the same underlying data the tiles are made from, under the
same ODbL licence, and it costs one Overpass query that caches to a gpkg.
It also gives control the tiles never did: the network can be drawn thinner
than the nodes, which is what a locator wants.

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
    ap.add_argument("--title", action="store_true",
                    help="draw the place name and node count in the image")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    default=True, help="re-query OSM rather than reading the "
                                       "cached basemap")
    args = ap.parse_args()
    name = CFG.get("study_area_name", "study area")
    banner(f"sampling nodes: {name}")

    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(WEB)
    x, y = nodes.geometry.x.values, nodes.geometry.y.values
    span = max(np.ptp(x), np.ptp(y))
    cxm, cym = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    h = span / 2 * (1 + args.pad * 2)
    print(f"{len(nodes)} nodes")

    fg = "#ffffff" if args.dark else "#111111"
    bg = "#0b0b0d" if args.dark else "#ffffff"

    fig, ax = plt.subplots(figsize=(9.6, 9.6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.set_xlim(cxm - h, cxm + h); ax.set_ylim(cym - h, cym + h)
    ax.set_aspect("equal"); ax.set_axis_off()

    # THE CACHE CAN POISON THE FIGURE. contextily stores whatever came back,
    # including a provider's "requires API key" placeholder, and then serves it
    # forever without another request -- the London map carried that watermark
    # across the whole basemap while every live tile fetched fine. --no-cache
    # bypasses it; deleting the directory is the cure once it has happened.
    # streets and water from OSM, cached so a re-run is offline
    import osmnx as ox
    from shapely.geometry import box as shapely_box
    ll = gpd.GeoSeries([shapely_box(cxm - h, cym - h, cxm + h, cym + h)],
                       crs=WEB).to_crs(4326).iloc[0]
    w, s_, e, n_ = ll.bounds
    cache = PROC / f"basemap_{int(args.pad*100)}.gpkg"
    if cache.exists() and args.cache:
        roads = gpd.read_file(cache, layer="roads").to_crs(WEB)
        try:
            water = gpd.read_file(cache, layer="water").to_crs(WEB)
        except Exception:
            water = None
    else:
        print("  querying OSM for the basemap ...")
        G = ox.graph_from_bbox((w, s_, e, n_), network_type="all",
                               simplify=True, retain_all=True)
        roads = ox.graph_to_gdfs(G, nodes=False)[["geometry"]].to_crs(WEB)
        try:
            water = ox.features_from_bbox((w, s_, e, n_),
                                          {"natural": "water",
                                           "waterway": "riverbank"})
            water = water[water.geometry.type.isin(
                ["Polygon", "MultiPolygon"])][["geometry"]].to_crs(WEB)
        except Exception as ex:
            print(f"  no water layer: {ex}")
            water = None
        roads.to_crs(4326).to_file(cache, layer="roads", driver="GPKG")
        if water is not None and len(water):
            water.to_crs(4326).to_file(cache, layer="water", driver="GPKG")

    if water is not None and len(water):
        water.plot(ax=ax, color="#16181d" if args.dark else "#e8eef4",
                   linewidth=0, zorder=1)
    roads.plot(ax=ax, color="#2b2f36" if args.dark else "#d5d5d5",
               linewidth=.55, zorder=2)
    print(f"  basemap: {len(roads)} road segments"
          + (f", {len(water)} water polygons" if water is not None else ""))

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

    # No title or count in the image: a locator goes on a slide or beside a
    # caption that names the place already, and burned-in text cannot be
    # edited or translated without regenerating the figure. --title puts both
    # lines back. The attribution stays, because the licence requires it.
    if args.title:
        ax.text(.03, .965, name, transform=ax.transAxes, color=fg,
                fontsize=19, va="top")
        ax.text(.03, .925, f"{len(nodes)} sampling nodes, {step:.0f} m apart "
                f"along each street", transform=ax.transAxes, color=fg,
                fontsize=11, va="top", alpha=.78)
    ax.text(.985, .012, "basemap drawn from OpenStreetMap (ODbL)",
            transform=ax.transAxes, color=fg, fontsize=7, ha="right", alpha=.55)

    out = args.out or (RES / "figures" /
                       f"node_map_{CFG.get('study_area_slug', 'area')}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=bg, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
