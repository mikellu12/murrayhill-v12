"""Flag which exported panoramas have a sidewalk shed in view.

One row per image in data/raw/svi_180, so a shed can be found rather than
merely counted. The label comes from DOB permits -- the city's own record of
every shed, scaffold and construction fence -- filtered to those live in the
capture month, not from a detector. The CLIPSeg scaffolding detector scores
AUC 0.55 against exactly this label and 0.51 inside the forward cone, so it
is a coin flip and is deliberately not used here.

The label is DIRECTIONAL, which is the whole point of doing this per image
rather than per node. A panorama covers 180 degrees, so a shed twenty metres
behind the camera is not in it. Each node appears twice in the export, once
per walk direction, and a shed will usually show in one and not the other.
A node-level flag would mark both and be wrong half the time.

What the flag does and does not mean:

  in_view=True   a live shed permit sits within the match radius AND inside
                 the forward half of the panorama. It is a CANDIDATE, not a
                 confirmation -- a permit's point is the building address,
                 so a shed wrapping a corner is one point on one frontage,
                 and a shed across a wide avenue can be in range while
                 barely visible.
  in_view=False  no permit in the cone. Sheds erected and removed without a
                 live permit, or signed off early, will be missed.

So this narrows 1,254 images to a few hundred worth looking at. It does not
replace looking at them. Confirmed by eye, the column becomes ground truth.

    .venv/Scripts/python tools/svi_scaffold_flag.py --out data/raw/svi_180
"""
import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import CFG, PROC, RAW, RES, banner
from export_svi_180 import _street_axis, _walks

UTM = 32618
FOV = CFG["directional"]["fov"]
RADIUS = CFG["dob"]["match_radius_m"]
# All four permit types, flagged separately. A sidewalk shed is the obvious
# one -- an overhead deck the pedestrian walks under -- but it is a minority
# of what is actually standing on a Manhattan street: supported scaffold
# (frame up the facade), suspended scaffold (swing stage), and construction
# fence together outnumber sheds here. They look different, occlude
# different parts of the visual field, and a study of the pedestrian realm
# cares about all of them, so none is folded into the others.
WORK_TYPES = {
    "shed": "Sidewalk Shed",
    "supported": "Supported Scaffold",
    "suspended": "Suspended Scaffold",
    "fence": "Construction Fence",
}
SHED_TYPES = list(WORK_TYPES.values())


def _live_permits(at: str) -> gpd.GeoDataFrame:
    """Permits of the tracked types standing in the capture month.

    A permit is ground truth for an image only if the structure was up the
    day the image was taken, so the window is issued-before and expires-after
    the capture month rather than merely "on file".
    """
    p = pd.read_csv(RAW / "dob_permits.csv", low_memory=False)
    lo, hi = pd.Timestamp(f"{at}-01"), pd.Timestamp(f"{at}-28")
    issued = pd.to_datetime(p.issued_date, errors="coerce")
    expired = pd.to_datetime(p.expired_date, errors="coerce")
    live = (issued <= hi) & (expired >= lo)
    p = p[live & p.work_type.isin(SHED_TYPES)].copy()
    return gpd.GeoDataFrame(
        p, geometry=gpd.points_from_xy(p.longitude.astype(float),
                                       p.latitude.astype(float)),
        crs=4326).to_crs(UTM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/svi_180"),
                    help="the exported panorama tree to describe")
    ap.add_argument("--at", default="2026-04", help="capture month of the imagery")
    ap.add_argument("--radius", type=float, default=RADIUS)
    args = ap.parse_args()
    banner("flag panoramas with a shed in view")

    permits = _live_permits(args.at)
    print(f"{len(permits)} permits of {SHED_TYPES} live at {args.at}")
    print(permits.work_type.value_counts().to_string())
    groups = [("", permits)] + [(k + "_", permits[permits.work_type.eq(v)])
                                for k, v in WORK_TYPES.items()]
    groups = [(k, g.geometry.x.values, g.geometry.y.values) for k, g in groups]

    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(UTM)
    nodes["folder"] = nodes.chain.str.split("#").str[0]
    at_node = {r.node_id: r.geometry for r in nodes.itertuples()}

    rows = []
    for jpg in sorted(args.out.rglob("*.jpg")):
        m = re.match(r"(\d+)_(n\d+)_([NESW])\.jpg", jpg.name)
        if not m:
            continue
        seq, nid, card = int(m.group(1)), m.group(2), m.group(3)
        rows.append({"street": jpg.parent.parent.name, "direction": jpg.parent.name,
                     "seq": seq, "node_id": nid, "cardinal": card,
                     "file": str(jpg.relative_to(args.out)).replace("\\", "/")})
    df = pd.DataFrame(rows)
    print(f"{len(df)} panoramas in {args.out}\n")

    # The bearing each image faces, rebuilt exactly as the export built it:
    # one axis per street from its node positions, two opposite walks.
    bearing = {}
    for street, g in nodes[nodes.node_id.isin(df.node_id)].groupby("folder"):
        axis = _street_axis(g.geometry.x.to_numpy(), g.geometry.y.to_numpy())
        for b, walk in _walks(axis):
            bearing[(street, walk)] = b

    out = []
    for r in df.itertuples():
        pt = at_node[r.node_id]
        b = bearing[(r.street, r.direction)]
        rec = {"bearing": round(b, 1)}
        for prefix, xs, ys in groups:
            dx, dy = xs - pt.x, ys - pt.y
            d = np.hypot(dx, dy)
            near = d <= args.radius
            in_view, nearest = False, (round(float(d.min()), 1) if len(d) else np.nan)
            if near.any():
                # Bearing to each nearby permit, folded to the signed offset
                # from where the camera looks; inside half the field of view
                # means it falls in the panorama.
                brg = np.degrees(np.arctan2(dx[near], dy[near])) % 360
                off = np.abs(((brg - b + 180) % 360) - 180)
                in_view = bool((off <= FOV / 2).any())
                nearest = round(float(d[near].min()), 1)
            rec[prefix + "in_view"] = in_view
            rec[prefix + "nearest_m"] = nearest
        out.append(rec)

    df = pd.concat([df, pd.DataFrame(out)], axis=1)
    dest = RES / "tables" / "svi_180_scaffold.csv"
    df.sort_values(["street", "direction", "seq"]).to_csv(dest, index=False)

    n = len(df)
    print(f"{'any tracked structure':<24}{df.in_view.sum():>5} of {n} ({df.in_view.mean():.0%})")
    for k, v in WORK_TYPES.items():
        c = df[k + "_in_view"]
        print(f"  {v:<22}{c.sum():>5} of {n} ({c.mean():.0%})")
    # A node flagged in one walk and not the other is the directional label
    # doing its job; if this were zero the cone test would be pointless.
    per = df.groupby("node_id").in_view.nunique()
    print(f"nodes where the two walks disagree: {(per == 2).sum()} of {len(per)}")
    print("\nby street, share of images with a shed in view:")
    by = df.groupby("street").in_view.agg(["mean", "size"]).sort_values("mean", ascending=False)
    for s, r in by.iterrows():
        print(f"  {s:<24}{r['mean']:>5.0%}  ({int(r['size'])} images)")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
