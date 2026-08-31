"""Build a London frame from the colleague's node export.

The Murray Hill frame is generated here by s01 from the OSM drive network. The
London one arrives ready-made, so this converts it into the same shape the rest
of the pipeline expects rather than adapting every downstream tool.

NODE IDS ARE RENUMBERED. The source uses street-slug ids -- abchurch_lane_001 --
and the pipeline matches `n\\d{5}` in a dozen places: the mast probes, the
segmentation join, the export filenames, the resume logic. Renumbering once
here is safer than loosening every pattern. The original id is kept as
`source_id` so anything traced back to the colleague's frame still can be.

Ordering is by street then sequence, so `n00000` upward walks each street in
order and a sorted listing of exported imagery reads as a walk rather than a
shuffle.

SPACING IS ALREADY CORRECT and must not be "fixed". The global nearest-
neighbour distance is 12.0 m, which looks like over-sampling against the
manuscript's 20 m -- but measured WITHIN a street it is 18.4 m against Murray
Hill's 20.0 m. The 12 m is nodes on different streets sitting close together,
which is what the City of London is: Abchurch Lane and King William Street are
metres apart. Thinning on global spacing would delete real coverage on adjacent
streets to fix a problem that does not exist.

    SIM_CONFIG=config_london.yaml .venv/Scripts/python tools/import_london_nodes.py \\
        --csv <nodes.csv>
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
from common import PROC, banner

UTM = 27700          # OSGB36 / British National Grid, metres
SEQ_WIDTH = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--gpkg", type=Path, default=None)
    args = ap.parse_args()
    banner("import the London frame")

    c = pd.read_csv(args.csv, encoding="utf-8-sig")
    need = {"node_id", "street_name", "lat", "lng"}
    missing = need - set(c.columns)
    if missing:
        sys.exit(f"source is missing {sorted(missing)}")
    print(f"{len(c)} source nodes, {c.street_name.nunique()} streets")

    # street then sequence: n00000 upward walks each street in order
    seq = c.seq_fwd if "seq_fwd" in c.columns else pd.Series(range(len(c)))
    c = c.assign(_seq=seq).sort_values(["street_name", "_seq"]).reset_index(drop=True)
    c["source_id"] = c.node_id
    c["node_id"] = [f"n{i:0{SEQ_WIDTH}d}" for i in range(len(c))]

    # folder-safe street label, matching the Murray Hill exports' convention
    c["folder"] = (c.street_name.str.lower()
                   .str.replace(r"[^a-z0-9]+", "_", regex=True)
                   .str.strip("_"))
    c["osm_name"] = c.street_name
    # s02 reads r.lon; the source names it lng. Aliased rather than renamed so
    # a row still matches the colleague's export column for column.
    c["lon"] = c.lng

    g = gpd.GeoDataFrame(c, geometry=[Point(x, y) for x, y in zip(c.lng, c.lat)],
                         crs=4326)
    m = g.to_crs(UTM)
    g["easting_m"], g["northing_m"] = m.geometry.x.values, m.geometry.y.values

    # every downstream stage reads these, and a missing column is a late crash
    for col, val in (("is_tunnel", 0.0), ("is_bridge", 0.0),
                     ("chain", None), ("chain_pos_m", np.nan),
                     ("street_segment", None), ("cleaned_street", None),
                     ("typology", "unclassified"), ("in_study", True)):
        if col not in g.columns:
            g[col] = g.folder if col in ("chain", "street_segment",
                                         "cleaned_street") else val

    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "nodes.gpkg"
    g.to_file(out, driver="GPKG")
    g.drop(columns="geometry").to_csv(PROC / "nodes.csv", index=False)
    print(f"\nwrote {out}")
    print(f"      {PROC / 'nodes.csv'}")

    from scipy.spatial import cKDTree
    xy = np.c_[m.geometry.x, m.geometry.y]
    t = cKDTree(xy)
    d, i = t.query(xy, k=12)
    st = g.street_name.values
    same = []
    for r in range(len(xy)):
        for k in range(1, 12):
            if st[i[r, k]] == st[r]:
                same.append(d[r, k])
                break
    print(f"\n  {len(g)} nodes, ids n00000-n{len(g)-1:05d}")
    print(f"  spacing within a street: median {np.median(same):.1f} m "
          f"(Murray Hill 20.0 m)")
    print(f"  extent {(xy[:,0].max()-xy[:,0].min())/1000:.2f} x "
          f"{(xy[:,1].max()-xy[:,1].min())/1000:.2f} km")
    print(f"\n  imagery to fetch: {len(g)} nodes x 4 headings = {len(g)*4} requests")
    print(f"  largest streets:")
    for s, n in g.street_name.value_counts().head(5).items():
        print(f"    {s:<28}{n:>4}")


if __name__ == "__main__":
    main()
