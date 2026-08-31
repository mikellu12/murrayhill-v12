"""Import an externally authored node set into the pipeline's frame schema.

The v13 frame arrives as one row per image with its own naming
(1st_avenue_001_N.jpg), no chain positions, and no typology. This converts it
to what s02 onward expect, without inventing anything the source did not say.

Three things it does NOT do, on purpose:

1. It does not renumber against the old frame. Node ids are positional, so a
   v13 n00042 is a different street corner from a v12 n00042. The old frame
   and its profiles are moved aside rather than overwritten, because the only
   safe way to read a profile is alongside the frame that produced it.

2. It does not merge Park Avenue's carriageways. The source separates
   Park_Ave_East, Park_Ave_West and Park_Ave_Tunnel_Segment, which is what
   fixes the v12 fold where chain_pos_m ran up one side and back down the
   other (Spearman 0.043 against physical position). They stay separate
   chains; only osm_name is unified, so typology still resolves.

3. It does not decide whether the tunnel segments belong in the study. The
   v12 frame excluded them by OSM tag; this one names them explicitly. That
   is a judgement about what a pedestrian streetscape is, not a data
   question, so they are imported and flagged for a human to rule on.

    .venv/Scripts/python tools/import_frame.py <source.gpkg> [--apply]
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, CFG, PROC, banner, typology_of, grid_northing, grid_easting

# Source category -> the name typology_of() matches against. Park Avenue's
# three chains share one street name so the canyon pattern still fires.
STREET = {
    "Park_Ave_East": "Park Avenue",
    "Park_Ave_West": "Park Avenue",
    "Park_Ave_Tunnel_Segment": "Park Avenue",
    "Tunnel_Exit_Street": "Tunnel Exit Street",
    "tudor_city_place": "Tudor City Place",
}
TUNNEL = {"Park_Ave_Tunnel_Segment", "Tunnel_Exit_Street"}


def street_name(cat):
    if cat in STREET:
        return STREET[cat]
    s = cat.replace("_", " ")
    s = re.sub(r"\beast\b", "East", s, flags=re.I)
    s = re.sub(r"\bstreet\b", "Street", s, flags=re.I)
    s = re.sub(r"\bavenue\b", "Avenue", s, flags=re.I)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: m.group(0).lower(), s)
    return s[:1].upper() + s[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--apply", action="store_true",
                    help="replace data/processed/nodes.gpkg (backs up the old frame first)")
    args = ap.parse_args()
    banner("import frame")

    src = gpd.read_file(args.source).to_crs(PROJ_CRS)
    key = "original_id" if "original_id" in src.columns else "image_id"
    n = src.drop_duplicates(key).copy().reset_index(drop=True)
    print(f"source rows {len(src)}  ->  unique nodes {len(n)}")

    n["osm_name"] = n.street_category.map(street_name)
    n["chain"] = n.street_category          # carriageways stay separate
    n["is_tunnel"] = n.street_category.isin(TUNNEL)

    # chain_pos_m from the source's own ordering, which the ids encode.
    n["_seq"] = n[key].str.extract(r"(\d+)\s*$").astype(int)
    out = []
    for ch, g in n.groupby("chain"):
        g = g.sort_values("_seq").copy()
        P = np.c_[g.geometry.x, g.geometry.y]
        step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
        g["chain_pos_m"] = np.cumsum(step)
        out.append(g)
    n = pd.concat(out).sort_values(["chain", "chain_pos_m"]).reset_index(drop=True)
    n = gpd.GeoDataFrame(n, geometry="geometry", crs=PROJ_CRS)

    n["node_id"] = [f"n{i:05d}" for i in range(len(n))]
    w = n.to_crs(4326)
    n["lat"], n["lon"] = w.geometry.y, w.geometry.x
    n["typology"] = typology_of(n.osm_name)
    n["northing_m"] = grid_northing(n.lat, n.lon)
    n["easting_m"] = grid_easting(n.lat, n.lon)

    cols = ["node_id", "osm_name", "chain", "chain_pos_m", "lat", "lon",
            "typology", "northing_m", "easting_m", "is_tunnel", "geometry"]
    n = n[cols]

    print("\n=== typology resolution ===")
    print(n.groupby(["typology", "osm_name"]).size().to_string())
    unres = sorted(n.loc[n.typology.isin(["other", "", None]), "osm_name"].unique())
    if unres:
        print(f"\n  unmatched by any pattern in config: {unres}")
        print("  -> add a pattern under study_area: or they carry no typology")

    print(f"\n=== tunnel segments (v12 excluded these by OSM tag) ===")
    print(n[n.is_tunnel].groupby("chain").size().to_string() or "  none")
    print(f"  {int(n.is_tunnel.sum())} of {len(n)} nodes. Kept and flagged; "
          "decide before running s02.")

    dst = PROC / "nodes_v13.gpkg"
    # Explicit layer name: GPKG defaults to the file stem, so writing
    # nodes_v13.gpkg then copying it to nodes.gpkg leaves a layer called
    # nodes_v13 as the default, and anything that later writes a "nodes"
    # layer adds a second one instead of replacing it.
    n.to_file(dst, driver="GPKG", layer="nodes")
    print(f"\nwrote {dst}  ({len(n)} nodes)")

    if args.apply:
        for f in ("nodes.gpkg", "azimuth_profiles.npz", "sim_profiles.npz",
                  "metrics.csv", "metrics.gpkg", "manifest.csv"):
            p = PROC / f
            if p.exists():
                shutil.move(str(p), str(PROC / f"v12_{f}"))
                print(f"  moved {f} -> v12_{f}")
        shutil.copy(str(dst), str(PROC / "nodes.gpkg"))
        print("  nodes.gpkg is now the v13 frame")
        print("\nnext: s02 needs GMAPS_KEY -- no imagery matches this frame.")


if __name__ == "__main__":
    main()
