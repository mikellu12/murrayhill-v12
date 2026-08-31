"""Label every node vehicular or pedestrian, so the render can match the view.

Street View is captured from a vehicle on the roadway, and that fixes where the
camera stands relative to what a walker sees. On a wide vehicular street the
camera is tens of metres from either frontage, so a 90-degree half centred on
each side faces one frontage squarely and at 16 px/degree. On a narrow
pedestrian way camera and walker share a position, both frontages are metres
away, and enclosure fills the visual field -- a half-view would cut out most of
what makes the space feel as it does, so the 180-degree strip is the honest
render.

The field of view is therefore part of what is measured, not a nuisance to
correct away. The consequence, which belongs in the methods: an M from a
180-degree walkway and an M from a 90-degree street are not strictly on one
scale and should be reported as typology-specific.

OSM's `highway` tag decides it. Matching is spatial and by nearest way, because
the node frame and the OSM graph are separately generated and do not share ids.
A node further than `--max-snap-m` from any way keeps the vehicular default
rather than being guessed at: the 90-degree render is the conservative choice,
since it is what everything downstream was validated on.

    SIM_CONFIG=config_london.yaml .venv/Scripts/python tools/classify_street_type.py
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, banner

# highway= values that describe a way people walk on and vehicles do not
# dominate. living_street is included: motor traffic is subordinate to
# pedestrians by design, which is the perceptual situation that matters here.
PEDESTRIAN = {"footway", "pedestrian", "steps", "path", "cycleway",
              "living_street", "corridor"}

# A footway is not necessarily a pedestrian STREET. Central London maps the
# pavement beside every road as its own way, so a node standing on the kerb of
# Cannon Street snaps to a footway and would be called pedestrian -- 527 nodes
# came out that way on the first pass, including four of the City's busiest
# roads. OSM separates them by subtag: 2,401 of 5,969 footways here are
# footway=sidewalk and 859 are crossings. What remains is the real thing --
# Change Alley, Peter's Hill, the Barbican highwalks.
NOT_A_STREET = {"sidewalk", "crossing", "traffic_island", "link"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-snap-m", type=float, default=12.0)
    ap.add_argument("--crs", type=int, default=27700)
    ap.add_argument("--margin-deg", type=float, default=0.004)
    args = ap.parse_args()
    banner("classify each node vehicular or pedestrian")

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    # The two frames do not agree on storage CRS -- London's is 4326, Murray
    # Hill's is 32618 -- and the OSM query needs degrees. Taking total_bounds
    # off a projected frame hands Overpass easting/northing, which surfaces as
    # a NaN deep inside osmnx rather than as a complaint about units.
    if nodes.crs is not None and nodes.crs.to_epsg() != 4326:
        print(f"nodes are {nodes.crs.to_string()}; reprojecting to 4326 "
              f"for the OSM query")
        nodes = nodes.to_crs(4326)
    b = nodes.total_bounds
    m = args.margin_deg
    bbox = (b[0] - m, b[1] - m, b[2] + m, b[3] + m)
    print(f"{len(nodes)} nodes, bbox {bbox[0]:.4f} {bbox[1]:.4f} "
          f"{bbox[2]:.4f} {bbox[3]:.4f}")

    import osmnx as ox
    ways = ox.features_from_bbox(bbox, {"highway": True})
    ways = ways[ways.geometry.type.isin(["LineString", "MultiLineString"])]
    ways = ways.reset_index(drop=True)
    ways["hw"] = ways.highway.astype(str)
    print(f"{len(ways)} highway ways downloaded\n")

    sub = (ways.footway.astype(str) if "footway" in ways.columns
           else pd.Series("", index=ways.index))
    drop = (ways.hw == "footway") & sub.isin(NOT_A_STREET)
    print(f"dropping {int(drop.sum())} sidewalk/crossing footways")
    ways = ways[~drop].copy()
    ways["wname"] = (ways.name.astype(str) if "name" in ways.columns else "")

    keep = ["hw", "wname", "geometry"]
    j = gpd.sjoin_nearest(nodes.to_crs(args.crs)[["node_id", "osm_name", "geometry"]],
                          ways.to_crs(args.crs)[keep],
                          how="left", distance_col="snap_m")
    # Prefer a way that carries the node's own street name over a merely closer
    # one. Proximity alone hands a Cannon Street node to whatever unnamed alley
    # runs behind it; the name is the stronger evidence of which street the node
    # is actually on, and distance only breaks ties.
    j["named"] = (j.wname.astype(str).str.casefold()
                  == j.osm_name.astype(str).str.casefold())
    j = (j.sort_values(["named", "snap_m"], ascending=[False, True])
           .groupby("node_id").first().reset_index())
    print(f"{int(j.named.sum())} of {len(j)} nodes matched a way of the same name")

    far = j.snap_m > args.max_snap_m
    j["is_pedestrian"] = j.hw.isin(PEDESTRIAN) & ~far
    j["fov"] = np.where(j.is_pedestrian, 180, 90)
    j.loc[far, "hw"] = "unmatched"

    print(f"snap distance: median {j.snap_m.median():.1f} m, "
          f"{int(far.sum())} beyond {args.max_snap_m:g} m (kept vehicular)\n")
    print("  highway= of the nearest way:")
    print("   " + j.hw.value_counts().to_string().replace("\n", "\n   "))
    n_ped = int(j.is_pedestrian.sum())
    print(f"\n  {n_ped} pedestrian nodes -> 180 degrees, "
          f"{len(j)-n_ped} vehicular -> two 90-degree halves")
    print(f"  images to render: {n_ped*2} + {(len(j)-n_ped)*4} = "
          f"{n_ped*2 + (len(j)-n_ped)*4}")
    if n_ped:
        print(f"\n  pedestrian streets:")
        s = j[j.is_pedestrian].osm_name.value_counts()
        for k, v in s.head(12).items():
            print(f"    {str(k)[:34]:<35}{v:>4}")
        if len(s) > 12:
            print(f"    ... and {len(s)-12} more")

    out = PROC / "street_type.csv"
    j[["node_id", "osm_name", "hw", "snap_m", "is_pedestrian", "fov"]].to_csv(
        out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
