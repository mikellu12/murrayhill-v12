"""Mark which nodes survive the hand-checked cleaning of the frame.

final_nodes_cleaned.gpkg is an externally authored review of the sampling
frame: someone went through it and removed what is not pedestrian streetscape.
It holds 712 of the frame's 766 locations, matching them to 0.00 m, so it is a
strict subset -- a set of decisions about the frame, not a different frame.

The 54 it removes are coherent rather than arbitrary:

    is_tunnel      36 of 54 dropped (67%) against 42 of 712 kept (5.9%)
    median GVI      1.187 dropped against 2.277 kept
    GVI exactly 0   8 of 54 (14.8%) against 22 of 712 (3.1%)

They are Park Avenue's tunnel segment, its surplus carriageways, and the four
viaduct nodes that export_svi_180.py already carries as a hardcoded list. And
it keeps Tunnel Exit Street, which is the same call study_area_filter.py makes
in prose: that street has one roadway, so it duplicates nothing.

WHY THIS IS A FLAG AND NOT A REIMPORT. node_id is positional. Re-importing the
cleaned file renumbers every node, and 2,940 image filenames, both profile
arrays, metrics.csv, sim_index.csv, the 1,254-image export and every table
built from them are keyed to the current ids. Adding a column changes nothing
anyone else reads. The cleaning is honoured, the invariant holds, and analyses
can filter on `in_cleaned` or compare with and without it.

Only 2 of the 54 appear in the 1,254-image export, so the flag is close to
free on the current results -- which is worth knowing before it is applied,
not after.

    .venv/Scripts/python tools/cleaned_frame_flag.py <cleaned.gpkg> [--apply]
"""
import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, banner

TOL_M = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="final_nodes_cleaned.gpkg")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tol", type=float, default=TOL_M,
                    help="metres within which a node counts as the same point")
    args = ap.parse_args()
    banner("flag nodes kept by the cleaned frame")

    if not args.source.exists():
        sys.exit(f"no such file: {args.source}")
    n = gpd.read_file(PROC / "nodes.gpkg").to_crs(PROJ_CRS)
    c = gpd.read_file(args.source).to_crs(PROJ_CRS)
    if "original_id" in c.columns:
        c = c.drop_duplicates("original_id")
    print(f"frame {len(n)} nodes   cleaned source {len(c)} locations")

    nx = np.column_stack([n.geometry.x, n.geometry.y])
    cx = np.column_stack([c.geometry.x, c.geometry.y])
    d = np.sqrt(((nx[:, None] - cx[None]) ** 2).sum(-1))
    near = d.min(1)
    n["in_cleaned"] = near < args.tol

    # Carry the cleaned file's street organisation across, which is the part
    # worth having. Tested by Spearman of sequence against position along each
    # street's principal axis, it orders 19 of 20 streets perfectly, where the
    # live chain_pos_m manages 15 of 19 -- Park_Ave_Tunnel_Segment sits at
    # 0.318 and Park_Ave_West at 0.474 even after repair_chain_pos.py ran.
    # The cleaned file avoids both by splitting the tangled chains into
    # 1st_avenue_west_branch and tunnel_approach_street.
    idx = d.argmin(1)
    for src, dst in (("street_category", "cleaned_street"),
                     ("direction", "cleaned_direction"),
                     ("original_id", "cleaned_id")):
        if src in c.columns:
            v = c[src].to_numpy()[idx]
            n[dst] = np.where(n.in_cleaned, v, None)

    # A strict subset is the premise. If the cleaned file holds points the
    # frame does not, it is a different frame and this tool is the wrong one.
    orphan = int((d.min(0) >= args.tol).sum())
    print(f"  matched within {args.tol} m: {int(n.in_cleaned.sum())} of {len(n)}")
    print(f"  max distance among matches : {near[n.in_cleaned].max():.3f} m")
    print(f"  cleaned points with no node in the frame: {orphan}")
    if orphan:
        sys.exit("cleaned file is not a subset of the frame -- refusing to flag")

    drop = n[~n.in_cleaned].copy()
    drop["st"] = drop.chain.str.split("#").str[0]
    print(f"\n{len(drop)} nodes dropped by the cleaning:")
    print(drop.groupby(["st", "is_tunnel"]).size().to_string())

    if "in_study" in n.columns:
        both = int((n.in_study & n.in_cleaned).sum())
        lost = int((n.in_study & ~n.in_cleaned).sum())
        gain = int((~n.in_study & n.in_cleaned).sum())
        print(f"\nagainst the existing study filter:")
        print(f"  in_study and in_cleaned : {both}")
        print(f"  in_study, not cleaned   : {lost}   (would be lost)")
        print(f"  cleaned, not in_study   : {gain}   (cleaning keeps these)")

    ex = HERE.parent / "results" / "tables" / "svi_180_scaffold.csv"
    if ex.exists():
        hit = pd.read_csv(ex).node_id.isin(set(drop.node_id)).sum()
        nodes_hit = pd.read_csv(ex).query("node_id in @drop.node_id").node_id.nunique()
        print(f"\nimpact on the 1,254-image export: {nodes_hit} nodes, {hit} images")

    if "cleaned_street" in n.columns:
        agree = n[n.in_cleaned].copy()
        agree["st"] = agree.chain.str.split("#").str[0]
        same = (agree.st == agree.cleaned_street).mean()
        print("")
        print(f"street label agrees with the live chain on "
              f"{same:.1%} of matched nodes")
        diff = agree[agree.st != agree.cleaned_street]
        if len(diff):
            print("  where they differ:")
            print(diff.groupby(["st", "cleaned_street"]).size().to_string())

    if not args.apply:
        print("\n(dry run -- pass --apply to write in_cleaned into nodes.gpkg)")
        return

    bak = PROC / "nodes.gpkg.bak"
    if not bak.exists():
        shutil.copy(PROC / "nodes.gpkg", bak)
        print(f"\nbacked up -> {bak}")
    n.to_crs(4326).to_file(PROC / "nodes.gpkg", driver="GPKG", layer="nodes")
    print(f"wrote in_cleaned into {PROC / 'nodes.gpkg'} "
          f"({int(n.in_cleaned.sum())} True, {int((~n.in_cleaned).sum())} False)")
    print("node_id and geometry are untouched; only a column was added.")


if __name__ == "__main__":
    main()
