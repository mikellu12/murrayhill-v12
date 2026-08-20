"""Sidewalk sheds and scaffolding from DOB permits, as ground truth.

WHY PERMITS RATHER THAN MORE PROMPTING
--------------------------------------
The open-vocabulary detector in tools/face_samples.py has no ground truth
behind it. Its threshold was set by looking at two frames, and an
untuned threshold produces a number that cannot be defended: on a plain
CLIPSeg score a distant glass facade on East 42nd read 11.8% scaffolding
with no shed anywhere in it.

New York happens to solve this. Every sidewalk shed, supported or
suspended scaffold, and construction fence needs a DOB permit, and DOB NOW
publishes each one with an address, a latitude and longitude, an issue date
and an expiry date. That gives a label that does not depend on what a model
thinks a photograph looks like.

The dates are what make it work here. A shed is only ground truth for an
image if it was standing the day the image was taken, and this frame's
imagery is filtered to a single capture month -- so "live at 2026-04" is a
sharp filter rather than a guess. Inside the study bbox 985 permits were
live that month: 395 sidewalk sheds, 237 supported and 248 suspended
scaffolds, 105 construction fences.

WHERE IT IS WRONG, AND WHY THAT IS STILL FINE
---------------------------------------------
A permit is filed against a building, so its point is the address, not the
structure -- a shed wrapping a corner is one point on one frontage. Permits
are signed off late often enough that a live record can outlast the shed.
And a shed across a wide avenue is within matching distance while being
barely visible from the node.

All three blur the label; none of them is correlated with what a CLIPSeg
prompt thinks. That is the property that matters: an imperfect label with
errors independent of the detector still measures the detector. It is a
tuning and validation set, not a second opinion to be averaged in.

    python tools/dob_sheds.py                # fetch, join, report
    python tools/dob_sheds.py --at 2026-04   # a different capture month
    python tools/dob_sheds.py --no-fetch     # re-join the cached pull

Needs network. Writes data/raw/dob_permits.csv (cached),
data/processed/dob_shed_by_node.csv and results/tables/dob_sheds.csv.
"""
import argparse, sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import CFG, PROC, RAW, RES, banner                 # noqa: E402

D = CFG["dob"]
API = f"https://data.cityofnewyork.us/resource/{D['dataset']}.json"
CACHE = RAW / "dob_permits.csv"


def fetch(bbox, at):
    """Every live permit of the tracked work types inside the bbox."""
    import requests
    lo, hi = f"{at}-01T00:00:00", f"{at}-28T23:59:59"
    types = ",".join(f"'{t}'" for t in D["work_types"])
    where = (f"work_type in({types}) "
             f"AND latitude between {bbox['south']} and {bbox['north']} "
             f"AND longitude between {bbox['west']} and {bbox['east']} "
             f"AND issued_date <= '{hi}' AND expired_date >= '{lo}'")
    rows, offset = [], 0
    while True:
        r = requests.get(API, params={"$where": where, "$limit": D["page_size"],
                                      "$offset": offset},
                         headers={"User-Agent": "murrayhill-gvi-research/1.0"},
                         timeout=120)
        r.raise_for_status()
        page = r.json()
        rows += page
        if len(page) < D["page_size"]:
            break
        offset += D["page_size"]
    d = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False)
    print(f"  fetched {len(d)} permits live at {at} -> {CACHE}")
    return d


def join(permits, nodes, radius):
    """Nearest live permit to each node, and the count within the radius."""
    from scipy.spatial import cKDTree
    p = gpd.GeoDataFrame(
        permits,
        geometry=gpd.points_from_xy(permits.longitude.astype(float),
                                    permits.latitude.astype(float)),
        crs=4326).to_crs(nodes.crs)
    tree = cKDTree(np.c_[p.geometry.x, p.geometry.y])
    xy = np.c_[nodes.geometry.x, nodes.geometry.y]
    dist, idx = tree.query(xy, k=1)
    within = tree.query_ball_point(xy, radius)

    out = nodes[["node_id", "osm_name"]].copy()
    out["dob_nearest_m"] = dist.round(1)
    out["dob_n_within"] = [len(w) for w in within]
    out["dob_shed"] = [any(p.work_type.iloc[j] == "Sidewalk Shed" for j in w)
                       for w in within]
    out["dob_any"] = out.dob_n_within > 0
    out["dob_types"] = ["|".join(sorted({p.work_type.iloc[j] for j in w}))
                        for w in within]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=CFG["capture"]["target"],
                    help="capture month, YYYY-MM (default from config)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="re-join the cached pull instead of querying")
    a = ap.parse_args()

    banner(f"DOB PERMITS  scaffolding live at {a.at}")
    if a.no_fetch and CACHE.exists():
        permits = pd.read_csv(CACHE)
        print(f"  cached: {len(permits)} permits from {CACHE}")
    else:
        permits = fetch(CFG["study_area"]["bbox"], a.at)
    if permits.empty:
        sys.exit("no permits returned -- check the bbox and the date")
    permits = permits.dropna(subset=["latitude", "longitude"])
    print("\n  by work type:")
    print("  " + permits.work_type.value_counts().to_string().replace("\n", "\n  "))

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    j = join(permits, nodes, D["match_radius_m"])
    j.to_csv(PROC / "dob_shed_by_node.csv", index=False)
    permits.to_csv(RES / "tables" / "dob_sheds.csv", index=False)

    r = D["match_radius_m"]
    print(f"\n  nodes within {r} m of a live permit: "
          f"{j.dob_any.sum()} of {len(j)} ({j.dob_any.mean():.0%})")
    print(f"  of a live SIDEWALK SHED specifically: {j.dob_shed.sum()} "
          f"({j.dob_shed.mean():.0%})")
    print("\n  by street, share of nodes with a shed within "
          f"{r} m:")
    by = (j.groupby("osm_name").dob_shed.agg(["mean", "size"])
           .sort_values("mean", ascending=False))
    for name, row in by.iterrows():
        print(f"    {name:20s} {row['mean']:5.0%}  ({int(row['size'])} nodes)")

    # Against the open-vocabulary detector, where samples exist.
    fs = RES / "tables" / "face_sample_shares.csv"
    if fs.exists():
        sh = pd.read_csv(fs)
        if "scaffolding" in sh.columns:
            m = sh.merge(j, on="node_id", how="left")
            print("\n  --- CLIPSeg detector against the permit label ---")
            print(f"  {'node':9s} {'street':18s} {'CLIPSeg %':>9s} "
                  f"{'permit':>8s} {'nearest m':>9s}")
            for _, x in m.sort_values("scaffolding", ascending=False).iterrows():
                print(f"  {x.node_id:9s} {str(x.osm_name_x)[:18]:18s} "
                      f"{100 * x.scaffolding:9.1f} "
                      f"{'SHED' if x.dob_shed else ('other' if x.dob_any else '-'):>8s} "
                      f"{x.dob_nearest_m:9.1f}")
            lab = m.dob_shed.fillna(False).astype(bool)
            if lab.nunique() > 1:
                a_, b_ = m.scaffolding[lab], m.scaffolding[~lab]
                print(f"\n  median CLIPSeg share where a shed is permitted: "
                      f"{100 * a_.median():.1f}% ({len(a_)} nodes)")
                print(f"  where none is:                                 "
                      f"{100 * b_.median():.1f}% ({len(b_)} nodes)")
                print("  25 sampled nodes is too few to set a threshold on."
                      " Run the\n  detector over every node, then tune"
                      " against this column.")
    print(f"\nwrote dob_shed_by_node.csv, dob_sheds.csv")


if __name__ == "__main__":
    main()
