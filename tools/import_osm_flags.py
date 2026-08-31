"""Attach OSM tunnel and bridge tags to the frame, from street-view-nodes.

This repository has three separate mechanisms for deciding what is a tunnel,
and they disagree with each other:

  s01              drops edges tagged bridge or tunnel -- correct, but only
                   at frame-build time, and the result is not kept per node
  exclude_pattern  a regex over the street NAME. It matches Tunnel Exit
                   Street and Tunnel Approach Street, which are ordinary
                   surface streets with sky and frontage, and it cannot see
                   the Park Avenue viaduct at all because nothing in that
                   name says bridge
  VIADUCT_NODES    four node IDs typed into export_svi_180.py by hand, plus
                   a post-segmentation sky-share heuristic that only fires
                   after the imagery has been fetched and segmented

street-view-nodes carries the OSM tags themselves, per node. Measured against
its output for this study area:

  flagged is_bridge, name says nothing   15 nodes, all Park Avenue -- the
                                         viaduct our hardcoded set gropes at
  named "tunnel", correctly NOT flagged  58 nodes across Tunnel Approach
                                         Street, Tunnel Exit Street and the
                                         named approach ramps

So the tags are better than the name in both directions: they find what the
name hides and spare what the name wrongly condemns.

Matching is spatial, not by name or ID -- the two frames sample the same
streets but phase their 20 m spacing differently, so nodes sit a median 2.5 m
apart. Anything beyond max_snap_m is left unflagged rather than guessed.

    .venv/Scripts/python tools/import_osm_flags.py --csv <their_nodes.csv>
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, banner

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
# seq_fwd/seq_rev order the nodes along each direction of travel. They were
# dropped for years, which forced the exporter to re-derive the walk ordering
# by projecting onto a fitted street axis -- adequate on a grid, wrong on a
# curving street. Carried through now so the frame supplies both the bearing
# and the order; see tools/export_svi_90.py._plan_street.
BRING = ["is_tunnel", "is_bridge", "heading_fwd_deg", "heading_rev_deg",
         "seq_fwd", "seq_rev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--max-snap-m", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    banner("import OSM tunnel/bridge tags onto the frame")

    t = pd.read_csv(args.csv, encoding="utf-8-sig")
    missing = [c for c in BRING if c not in t.columns]
    if missing:
        sys.exit(f"source is missing {missing}")
    src = gpd.GeoDataFrame(
        t, geometry=[Point(x, y) for x, y in zip(t.lng, t.lat)],
        crs=4326).to_crs(UTM)
    print(f"{len(src)} source nodes, {int(src.is_tunnel.sum())} tunnel, "
          f"{int(src.is_bridge.sum())} bridge")

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    j = gpd.sjoin_nearest(nodes.to_crs(UTM)[["node_id", "geometry"]],
                          src[BRING + ["geometry"]],
                          how="left", distance_col="snap_m")
    far = j.snap_m > args.max_snap_m
    print(f"snap distance: median {j.snap_m.median():.2f} m, "
          f"max {j.snap_m.max():.2f} m")
    print(f"{int(far.sum())} node(s) beyond {args.max_snap_m:g} m -- left unflagged")

    # A node with no counterpart is UNKNOWN, not "not a tunnel". Leaving it
    # NaN keeps that distinction; coercing to False would quietly assert the
    # thing we could not check.
    lab = j[["node_id"] + BRING].copy()
    for c in ("is_tunnel", "is_bridge"):
        lab[c] = np.where(far, np.nan, lab[c]).astype("float")
    for c in ("heading_fwd_deg", "heading_rev_deg"):
        lab[c] = np.where(far, np.nan, lab[c]).astype("float")
    lab = lab.rename(columns={"heading_fwd_deg": "osm_heading_fwd",
                              "heading_rev_deg": "osm_heading_rev"})

    n_t = int(np.nansum(lab.is_tunnel))
    n_b = int(np.nansum(lab.is_bridge))
    print(f"\nof our {len(nodes)} nodes: {n_t} tunnel, {n_b} bridge, "
          f"{int(lab.is_tunnel.isna().sum())} unknown")
    if n_t or n_b:
        seg = nodes.get("street_segment")
        if seg is not None:
            f = lab.assign(seg=seg.values)
            for c, name in (("is_tunnel", "tunnel"), ("is_bridge", "bridge")):
                s = f[f[c] == 1].seg.value_counts()
                if len(s):
                    print(f"\n  {name} nodes by segment:")
                    print("   " + s.to_string().replace("\n", "\n   "))

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    cols = [c for c in lab.columns if c != "node_id"]
    for name, path in (("nodes.gpkg", PROC / "nodes.gpkg"),
                       ("metrics.gpkg", PROC / "metrics.gpkg")):
        if not path.exists():
            continue
        g = gpd.read_file(path)
        before = g.crs                     # s01 writes these in UTM, not 4326
        g = g.drop(columns=[c for c in cols if c in g.columns])
        g = g.merge(lab, on="node_id", how="left")
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=before)
        g.to_file(path, driver="GPKG")
        print(f"wrote {name}")

    mp = PROC / "metrics.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m = m.drop(columns=[c for c in cols if c in m.columns])
        m.merge(lab, on="node_id", how="left").to_csv(mp, index=False)
        print("wrote metrics.csv")


if __name__ == "__main__":
    main()
