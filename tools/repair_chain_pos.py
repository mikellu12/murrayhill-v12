"""Recompute chain_pos_m from geometry, leaving node_id untouched.

import_frame.py took chain_pos_m from the source's id numbering, on the
assumption that 1st_avenue_001, _002, _003 run along the street. For the
cross-streets they do. For 1st Avenue and Park Avenue's three chains they do
not, and street_axis() -- which reads the bearing between chain-adjacent
nodes -- then measured the angle between two nodes that are not neighbours.
Park_Ave_Tunnel_Segment came out at a median 117 deg, an east-west bearing
on a north-south avenue, which silently rotates the 180 deg window every
metric is integrated over.

Ordering is a greedy nearest-neighbour walk from the end of the chain
furthest from its centroid, not a projection onto a principal axis. Both
work for a straight run; only the walk survives a chain that curves, and the
tunnel segment does.

node_id is NOT reassigned. It is already written into 2,940 image filenames
and keyed into both profile arrays, so renumbering here would silently pair
every node with another node's imagery -- the exact failure the frame
invariant exists to prevent.

    .venv/Scripts/python tools/repair_chain_pos.py [--apply]
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

SPLIT_M = 60.0   # a step longer than this is a discontinuity, not a stride


def walk_order(P):
    """Greedy nearest-neighbour path from the most peripheral point."""
    n = len(P)
    if n < 3:
        return np.arange(n)
    start = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
    order, seen = [start], {start}
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[order[-1]], axis=1)
        d[list(seen)] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt); seen.add(nxt)
    return np.array(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    banner("repair chain_pos_m")

    n = gpd.read_file(PROC / "nodes.gpkg")
    before = n.set_index("node_id").chain_pos_m.copy()
    out = []
    for ch, g in n.groupby("chain"):
        g = g.copy()
        P = np.c_[g.geometry.x, g.geometry.y]
        o = walk_order(P)
        g = g.iloc[o].copy()
        step = np.r_[0.0, np.linalg.norm(np.diff(P[o], axis=0), axis=1)]
        # A jump means the walk left one continuous run and started another --
        # 1st Avenue and Park Avenue each carry parallel service roadways that
        # are not reachable along the kerb. street_axis() reads the bearing
        # between chain-adjacent nodes, so a join produces a bearing across the
        # gap rather than along the street. Split so every chain is one run.
        part = np.cumsum(step > SPLIT_M)
        g["chain"] = [ch if k == 0 else f"{ch}#{k}" for k in part]
        pos = []
        for k in np.unique(part):
            sub = step[part == k].copy(); sub[0] = 0.0
            pos.append(np.cumsum(sub))
        g["chain_pos_m"] = np.concatenate(pos)
        nsub = g.chain.nunique()
        print(f"  {ch:<26} n={len(g):>3}  median step={np.median(step[1:]):5.1f} m"
              + (f"  ->  split into {nsub} runs" if nsub > 1 else ""))
        out.append(g)

    n2 = gpd.GeoDataFrame(pd.concat(out), geometry="geometry", crs=n.crs)
    moved = (n2.set_index("node_id").chain_pos_m - before).abs()
    print(f"\n  chain_pos_m changed for {int((moved > 1).sum())} of {len(n2)} nodes")
    print(f"  node_id assignments unchanged: "
          f"{set(n.node_id) == set(n2.node_id)}")

    if args.apply:
        n2 = n2.sort_values(["chain", "chain_pos_m"]).reset_index(drop=True)
        (PROC / "nodes.gpkg").unlink()          # replace, never add a layer
        n2.to_file(PROC / "nodes.gpkg", driver="GPKG", layer="nodes")
        print(f"\nwrote {PROC / 'nodes.gpkg'}")
        print("next: re-run s04 so street_axis_deg is recomputed")
    else:
        print("\n(dry run -- pass --apply to write)")


if __name__ == "__main__":
    main()
