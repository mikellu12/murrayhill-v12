"""The cross-streets as a matrix: 9 streets by 5 blocks, laid out as the map.

Rows run East 42nd at the top down to East 34th at the bottom, columns run
Madison-Park on the left across to 2nd-1st on the right. So the table is
oriented the way the neighbourhood is: north at the top, west at the left,
and a cell sits where its block sits.

Only the six avenues cut a cross-street here. Tunnel Exit Street and Tudor
City Place are excluded as boundaries -- both are partial north-south runs
that cross some cross-streets and not others, so including them gave East
39th seven blocks and East 34th six, and no rectangular matrix exists over
a ragged set of boundaries. They remain in the frame and in every other
count; they are simply not treated as block edges.

The avenues cannot join this matrix: nine cross-streets cut them into eight
blocks, not five, so they need their own 6 x 8 table.

    .venv/Scripts/python tools/block_matrix.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

CROSS_M = 45.0
# West to east. These six, and only these six, bound a block.
AVENUES = ["Madison Avenue", "Park Avenue", "Lexington Avenue",
           "3rd Avenue", "2nd Avenue", "1st Avenue"]
SHORT_AV = {"Madison Avenue": "Madison", "Park Avenue": "Park",
            "Lexington Avenue": "Lex", "3rd Avenue": "3rd",
            "2nd Avenue": "2nd", "1st Avenue": "1st"}
# North to south.
STREETS = [f"East {n} Street" for n in
           ["42nd", "41st", "40th", "39th", "38th", "37th", "36th", "35th", "34th"]]
SHORT_ST = {s: s.replace("East ", "").replace(" Street", "") for s in STREETS}


def main():
    banner("block matrix: cross-streets x blocks")
    n = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in n.columns:
        n = n[n.in_study].copy()
    sim = pd.read_csv(PROC / "sim_index.csv")[["node_id", "SIM"]]
    n = n.merge(sim, on="node_id", how="left")

    xy = {s: np.c_[g.geometry.x, g.geometry.y] for s, g in n.groupby("osm_name")}
    counts = pd.DataFrame(index=[SHORT_ST[s] for s in STREETS],
                          columns=range(1, 6), dtype=float)
    means = counts.copy()
    labels = []

    for st in STREETS:
        g = n[n.osm_name == st]
        t = g.easting_m.values                      # west to east
        P = xy[st]
        cuts = []
        for av in AVENUES:
            Q = xy.get(av)
            if Q is None:
                continue
            d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
            if d.min() <= CROSS_M:
                i = int(np.unravel_index(d.argmin(), d.shape)[0])
                cuts.append((t[i], SHORT_AV[av]))
        cuts.sort()
        edges = [c[0] for c in cuts]
        if len(edges) != 6:
            print(f"  WARNING {st}: {len(edges)} avenue crossings, expected 6")
        idx = np.searchsorted(edges, t, side="right")
        for b in range(1, 6):                       # block b lies between cut b-1 and b
            sel = idx == b
            counts.loc[SHORT_ST[st], b] = int(sel.sum())
            v = g.SIM.values[sel]
            means.loc[SHORT_ST[st], b] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        if not labels:
            labels = [f"{cuts[b-1][1]}-{cuts[b][1]}" for b in range(1, 6)]

    # Number the columns rather than naming the bounding avenues: the block
    # id is 34th_1 .. 34th_5, so the header should read the same way. The
    # avenue pairs are printed once below instead of nine times across.
    counts.columns = range(1, 6)
    means.columns = range(1, 6)
    print("\ncolumns: " + ",  ".join(f"{i+1} = {l}" for i, l in enumerate(labels)))
    print("\n=== node counts ===")
    print(counts.astype(int).to_string())
    print("\n=== mean SIM ===")
    print(means.round(3).to_string())
    print(f"\ntotal nodes in matrix: {int(counts.values.sum())}")
    counts.astype(int).to_csv(RES / "tables" / "block_matrix_counts.csv")
    means.round(4).to_csv(RES / "tables" / "block_matrix_sim.csv")
    print(f"wrote block_matrix_counts.csv and block_matrix_sim.csv")


if __name__ == "__main__":
    main()
