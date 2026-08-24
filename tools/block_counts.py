"""Node counts per street and per street block.

A block here is the stretch between two consecutive crossings, which is the
unit the morphology argument is actually about -- East 41st between Madison
and Park is one built condition, and between Park and Lexington is another.
The frame's chains do not encode that: a chain runs the whole length of a
street and crosses five avenues on the way.

Crossings are found from the node geometry rather than from OSM, because the
frame has already been filtered and a crossing with a street that is not in
the study area is not a boundary for this purpose.

Ordering follows how the grid is read: cross-streets west to east, avenues
north to south. Blocks are numbered consecutively over the ones that hold
nodes, so a missing stretch does not leave a gap in the numbering -- Tunnel
Exit Street yields 4 blocks, not 6 with two empty.

    .venv/Scripts/python tools/block_counts.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

CROSS_M = 45.0          # two streets closer than this cross
SHORT = {"East 34th Street": "34th", "East 35th Street": "35th",
         "East 36th Street": "36th", "East 37th Street": "37th",
         "East 38th Street": "38th", "East 39th Street": "39th",
         "East 40th Street": "40th", "East 41st Street": "41st",
         "East 42nd Street": "42nd", "Madison Avenue": "Madison",
         "Park Avenue": "Park", "Lexington Avenue": "Lex",
         "3rd Avenue": "3rd", "2nd Avenue": "2nd", "1st Avenue": "1st",
         "Tudor City Place": "TudorCity", "Tunnel Exit Street": "TunnelExit"}


def is_avenue(name):
    return "Avenue" in name or name in ("Tudor City Place", "Tunnel Exit Street")


def main():
    banner("street and block node counts")
    n = gpd.read_file(PROC / "nodes.gpkg")
    m = pd.read_csv(PROC / "metrics.csv")
    n["analytic"] = n.node_id.isin(m.node_id)
    if "in_study" in n.columns:
        outside = int((~n.in_study).sum())
        n = n[n.in_study].copy()
        print(f"study-area filter: dropped {outside} nodes outside the area")
    xy = {s: np.c_[g.geometry.x, g.geometry.y] for s, g in n.groupby("osm_name")}

    # ---- per street
    per = (n.groupby("osm_name")
             .agg(nodes=("node_id", "size"), analytic=("analytic", "sum"),
                  typology=("typology", "first"))
             .reset_index())
    per["excluded"] = per.nodes - per.analytic
    per = per.sort_values(["typology", "osm_name"])
    print("=== nodes per street ===")
    print(per.to_string(index=False))
    per.to_csv(RES / "tables" / "nodes_per_street.csv", index=False)

    # ---- per block
    rows = []
    for st, g in n.groupby("osm_name"):
        # Read along the grid: avenues north to south, cross-streets west to east.
        t = -g.northing_m.values if is_avenue(st) else g.easting_m.values
        P = xy[st]
        cuts = []
        for other, Q in xy.items():
            if other == st or is_avenue(other) == is_avenue(st):
                continue
            d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
            if d.min() > CROSS_M:
                continue
            i = int(np.unravel_index(d.argmin(), d.shape)[0])
            cuts.append((t[i], other))
        cuts.sort()
        edges = [c[0] for c in cuts]
        names = [SHORT.get(c[1], c[1]) for c in cuts]
        idx = np.searchsorted(edges, t, side="right")

        # A fragment beyond the outermost crossing is the tail of the study
        # area, not a block: East 42nd carries one node past 1st Avenue.
        # Fold those into the neighbouring block rather than numbering them,
        # so a "block" always means a stretch bounded by two crossings.
        present = sorted(set(idx))
        if len(present) > 1:
            if (idx == present[0]).sum() < 3:
                idx[idx == present[0]] = present[1]
            present = sorted(set(idx))
        if len(present) > 1:
            if (idx == present[-1]).sum() < 3:
                idx[idx == present[-1]] = present[-2]

        seen = 0
        for b in sorted(set(idx)):
            sel = idx == b
            if not sel.sum():
                continue
            seen += 1
            lo = names[b - 1] if b > 0 else "start"
            hi = names[b] if b < len(names) else "end"
            rows.append({"block": f"{SHORT.get(st, st)}_{seen}",
                         "street": st, "between": f"{lo} - {hi}",
                         "nodes": int(sel.sum()),
                         "analytic": int(g.analytic.values[sel].sum())})
    blocks = pd.DataFrame(rows)
    blocks["excluded"] = blocks.nodes - blocks.analytic
    print(f"\n=== {len(blocks)} blocks ===")
    print(blocks.to_string(index=False))
    blocks.to_csv(RES / "tables" / "nodes_per_block.csv", index=False)
    print(f"\ntotals: {blocks.nodes.sum()} nodes in blocks "
          f"(frame has {len(n)})   median {blocks.nodes.median():.0f} nodes/block")


if __name__ == "__main__":
    main()
