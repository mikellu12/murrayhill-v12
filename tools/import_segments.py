"""Adopt the colleague's coordinate mapping as the street-segment label.

final_coordinates_mapping.csv is the same frame this pipeline already uses --
766 nodes, 20 m spacing, every point matching ours to 0.0 m -- under a
different naming scheme, plus the Cartesian B(i,j) grid the manuscript's
Figure 6 describes. Two things in it are worth adopting.

FIRST, it splits Park Avenue three ways: Park_Ave_West, Park_Ave_East and
Park_Ave_Tunnel_Segment. We treat all 132 Park Avenue nodes as one street,
which makes it the widest donor pool in the study area -- W spans 12.6 to
113.5 m and H/W spreads 3.81, because the tunnel approach sits in the same
pool as the boulevard. The segment fill then averages across a morphological
break. Splitting at the tunnel is the boundary `face_id` failed to provide;
face_id is 27 groups of ~33 nodes and is effectively street-level.

SECOND, `in_cleaned` excludes 50 Park Avenue nodes that this mapping keeps,
45 of which already have cached panoramas. They are recoverable for nothing.

The folder names under data/raw/svi_90 are deliberately NOT renamed. Only two
differ -- 1st_avenue_west_branch and tunnel_approach_street -- and renaming
them would orphan 120 already-rated images against sim_vlm's `file` key for no
analytical gain. The label is what the donor pools and the reporting read; the
path is just where the pixels live.

Matching is spatial and exact, not by name: nearest neighbour with a hard cap,
because a name join would silently mis-pair anything the two schemes disagree
about, which is the whole reason this file is interesting.

    .venv/Scripts/python tools/import_segments.py --csv <path>
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, banner

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
MAX_SNAP_M = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    banner("import street-segment labels from the coordinate mapping")

    c = pd.read_csv(args.csv).drop_duplicates("original_node_id")
    src = gpd.GeoDataFrame(
        c.copy(), geometry=[Point(x, y) for x, y in zip(c.lng, c.lat)],
        crs=4326).to_crs(UTM)
    print(f"{len(src)} nodes in the mapping, "
          f"{src.street_category.nunique()} segments")

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    j = gpd.sjoin_nearest(
        nodes.to_crs(UTM)[["node_id", "geometry"]],
        src[["original_node_id", "street_category", "cross_dist",
             "along_dist", "geometry"]],
        how="left", distance_col="snap_m")

    # A frame that matches to 0.0 m everywhere is the claim being tested here,
    # so a snap beyond a couple of metres means the two frames are NOT the same
    # and every label after it would be silently wrong.
    bad = j[j.snap_m > MAX_SNAP_M]
    if len(bad):
        sys.exit(f"{len(bad)} node(s) snap further than {MAX_SNAP_M} m "
                 f"(worst {bad.snap_m.max():.1f} m) -- frames differ, aborting")
    print(f"all {len(j)} nodes snap within {j.snap_m.max():.3f} m\n")

    keep = ["node_id", "street_category", "cross_dist", "along_dist",
            "original_node_id"]
    lab = j[keep].rename(columns={"street_category": "street_segment",
                                  "original_node_id": "mapping_id"})
    print("nodes per segment:")
    print(lab.street_segment.value_counts().to_string())

    old = nodes.get("cleaned_street")
    if old is not None:
        m = lab.merge(nodes[["node_id", "cleaned_street"]], on="node_id")
        d = m[m.cleaned_street.notna() & (m.cleaned_street != m.street_segment)]
        print(f"\n{len(d)} node(s) relabelled from cleaned_street:")
        print(d.groupby(["cleaned_street", "street_segment"]).size().to_string())
        n_new = int(m.cleaned_street.isna().sum())
        print(f"{n_new} node(s) had no cleaned_street and now have a segment")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    for name, path in (("nodes.gpkg", PROC / "nodes.gpkg"),
                       ("metrics.gpkg", PROC / "metrics.gpkg")):
        if not path.exists():
            continue
        g = gpd.read_file(path)
        before = g.crs                    # s01 writes these in UTM, not 4326
        g = g.drop(columns=[c for c in lab.columns if c != "node_id"
                            and c in g.columns])
        g = g.merge(lab, on="node_id", how="left")
        # Adding a label must not move the geometry. Writing 4326 back over a
        # 32618 file leaves coordinates that look plausible and silently break
        # every distance downstream -- s04 stamps the CRS on rather than
        # reprojecting, so degrees get treated as metres.
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=before)
        if g.crs != before:
            sys.exit(f"{name}: CRS would change {before} -> {g.crs}, refusing")
        g.to_file(path, driver="GPKG")
        print(f"wrote {name}  ({g.street_segment.notna().sum()} labelled, "
              f"CRS {g.crs.to_string()})")

    mp = PROC / "metrics.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m = m.drop(columns=[c for c in lab.columns if c != "node_id"
                            and c in m.columns])
        m.merge(lab, on="node_id", how="left").to_csv(mp, index=False)
        print(f"wrote metrics.csv")


if __name__ == "__main__":
    main()
