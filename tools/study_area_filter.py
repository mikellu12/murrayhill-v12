"""Mark which nodes are inside the defined study area.

Two kinds of node are in the frame but outside the area the study is about,
and both would otherwise be counted:

1. Park Avenue's surplus roadways. The rule is one sampled roadway per
   street, not an exclusion of tunnels. The v13 frame samples Park Avenue
   three times -- Park_Ave_East, Park_Ave_West and Park_Ave_Tunnel_Segment
   -- so a Park Avenue block holds 13 nodes where Madison holds 4: the same
   street entered once per carriageway. Keeping only Park_Ave_East makes it
   comparable with every other avenue.

   Tunnel Exit Street is deliberately NOT dropped, even though it is also a
   tunnel approach. It is its own street with one roadway, so it duplicates
   nothing; whether it belongs in the study is a separate question about
   what counts as pedestrian streetscape, and this filter does not answer
   that one.

2. Nodes past the outermost crossing on their street. East 42nd carries one
   node east of 1st Avenue and the avenues carry one north of 42nd. They sit
   beyond the boundary streets that define the area, so they belong to no
   block and have no matching condition on the other side.

Nothing is deleted. in_study is a column, so the excluded nodes stay
inspectable and the decision stays reversible.

    .venv/Scripts/python tools/study_area_filter.py [--apply]
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

CROSS_M = 45.0
DROP_CHAINS = ("Park_Ave_West", "Park_Ave_Tunnel_Segment")


def is_avenue(name):
    return "Avenue" in name or name in ("Tudor City Place", "Tunnel Exit Street")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    banner("study area filter")

    n = gpd.read_file(PROC / "nodes.gpkg")
    n["in_study"] = True

    # --- 1. one carriageway of Park Avenue
    dup = n.chain.str.startswith(DROP_CHAINS)
    n.loc[dup, "in_study"] = False
    print("=== Park Avenue: keeping one carriageway ===")
    print(n[n.osm_name == "Park Avenue"].groupby("chain").in_study
            .agg(nodes="size", kept="sum").to_string())

    # --- 2. nodes beyond the outermost crossing on their street
    xy = {s: np.c_[g.geometry.x, g.geometry.y] for s, g in n.groupby("osm_name")}
    edge = []
    for st, g in n.groupby("osm_name"):
        t = -g.northing_m.values if is_avenue(st) else g.easting_m.values
        P = xy[st]
        cuts = []
        for other, Q in xy.items():
            if other == st or is_avenue(other) == is_avenue(st):
                continue
            d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
            if d.min() <= CROSS_M:
                cuts.append(t[int(np.unravel_index(d.argmin(), d.shape)[0])])
        if not cuts:
            continue
        outside = (t < min(cuts)) | (t > max(cuts))
        if outside.any():
            edge.extend(g.node_id.values[outside])
    n.loc[n.node_id.isin(edge), "in_study"] = False
    print(f"\n=== beyond the boundary streets: {len(edge)} nodes ===")
    print(n[n.node_id.isin(edge)].groupby("osm_name").size().to_string())

    print(f"\n=== result ===")
    print(f"  frame            {len(n)}")
    print(f"  in study area    {int(n.in_study.sum())}")
    print(f"  excluded         {int((~n.in_study).sum())}")
    print(n[n.in_study].groupby("typology").size().to_string())

    if args.apply:
        (PROC / "nodes.gpkg").unlink()
        n.to_file(PROC / "nodes.gpkg", driver="GPKG", layer="nodes")
        print(f"\nwrote in_study to {PROC / 'nodes.gpkg'}")
    else:
        print("\n(dry run -- pass --apply to write)")


if __name__ == "__main__":
    main()
