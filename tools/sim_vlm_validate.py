"""Score every VLM rating against the quantity measured over its own arc.

Five of the nine fields have a measured counterpart in the azimuthal profiles.
Those five are the whole evidential basis for the other four, which have no
measurable analogue at all and between them carry the larger share of the
index -- facade_variation alone is 22 per cent of M.

Each half-view is scored against the share over the exact 90 degrees it was
rendered from, not against the node. Scoring a quarter-arc rating against a
whole-node GVI marks correct answers wrong: a half facing a blank wall while
the trees stand opposite should rate 1, and the node-level target counted
those trees. Measured earlier on the same ratings, that one change moved
green_eye_level from +0.537 to +0.787.

Intervals are bootstrapped clustered on face_id. Moran's I is 0.62-0.66 here
and both halves of a node share a profile, so an unclustered interval would be
far too narrow.

    .venv/Scripts/python tools/sim_vlm_validate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner, bin_mask
from sim_fields import FIELDS, JUDGEMENT_ONLY
from half_target import walk_bearings, SIDE_OFF

# VLM field -> (profile row, human label). Rows are vegetation, sky, building.
TWIN = {
    "vertical_greenery": (0, "vegetation share"),
    "green_eye_level": (0, "vegetation share"),
    "vertical_hardscape": (2, "building share"),
    "sky_openness": (1, "sky share"),
}
# green_softening is NOT in TWIN. GMI is "how far greenery relieves the
# enclosure", not a vegetation share, and sim_fields.py records it as having no
# measured counterpart. It is reported here as a related check only: a high
# correlation with vegetation share says the rating rises with greenery, which
# it should, but it does not validate the softening judgement itself.
RELATED = {"green_softening": (0, "vegetation share (related, not a twin)")}
RNG = np.random.default_rng(0)


def spearman_ci(x, y, groups, n=3000):
    s = pd.DataFrame({"x": x, "y": y, "g": groups}).dropna()
    if len(s) < 25 or s.x.nunique() < 2:
        return None
    r = s.x.corr(s.y, method="spearman")
    uniq = pd.unique(s.g)
    idx = {q: np.flatnonzero(s.g.to_numpy() == q) for q in uniq}
    bs = []
    for _ in range(n):
        sub = s.iloc[np.concatenate([idx[q] for q in RNG.choice(uniq, len(uniq))])]
        if sub.x.nunique() > 1:
            bs.append(sub.x.corr(sub.y, method="spearman"))
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(s)


def main():
    # THE TABLE IS AN ARGUMENT. Two copies of this tool existed, identical
    # except that one named sim_vlm.csv and the other sim_vlm_v2.csv in the
    # code -- and by the time anyone looked, the study had moved on to a third
    # table and both were validating runs nobody used. A --table argument is
    # the reason a second copy is never needed again.
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "sim_vlm_180_placeless.csv",
                    help="ratings table to validate; default is the current run")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    banner("VLM ratings against the measured arc")
    print(f"ratings: {args.table}")
    d = pd.read_csv(args.table)
    z = np.load(PROC / "azimuth_profiles.npz")
    prof = {k: z[k] for k in z.files}
    bear = walk_bearings()

    parts = d.file.str.split("/", expand=True)
    d["street_dir"], d["walk_dir"] = parts[0], parts[1]
    cols = {0: "arc_veg", 1: "arc_sky", 2: "arc_bld"}
    acc = {v: [] for v in cols.values()}
    miss = 0
    for r in d.itertuples():
        p = prof.get(r.node_id)
        b = bear.get((r.street_dir, r.walk_dir))
        if p is None or b is None or r.side not in SIDE_OFF:
            for v in acc.values():
                v.append(np.nan)
            miss += 1
            continue
        m = bin_mask((b + SIDE_OFF[r.side]) % 360, 90.0)
        W = p[3][m].sum()
        for i, name in cols.items():
            acc[name].append(p[i][m].sum() / W if W > 0 else np.nan)
    for k, v in acc.items():
        d[k] = v

    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "face_id"]]
    d = d.merge(met, on="node_id", how="left")
    g = d.face_id.fillna(d.node_id).to_numpy()
    print(f"{len(d)} half-views, {d.arc_veg.notna().sum()} matched to a profile arc"
          + (f", {miss} unmatched" if miss else ""))
    print(f"{d.face_id.nunique()} block faces for clustering\n")

    rows = []
    print(f"  {'field':<23}{'vs measured':<19}{'rho':>7}   {'95% CI (faces)':<20}"
          f"{'mode%':>7}")
    for f, (row, label) in list(TWIN.items()) + list(RELATED.items()):
        got = spearman_ci(d[f].to_numpy(float), d[cols[row]].to_numpy(float), g)
        v = d[f].dropna()
        mo = (v == v.mode().iloc[0]).mean() * 100
        if not got:
            print(f"  {f:<23}{label:<19}{'not computable':>7}")
            continue
        r, lo, hi, n = got
        print(f"  {f:<23}{label:<19}{r:>+7.3f}   [{lo:+.3f},{hi:+.3f}]  {mo:>6.0f}%")
        rows.append({"field": f, "twin": label, "rho": r, "lo": lo, "hi": hi,
                     "n": n, "mode_pct": mo})

    print(f"\n  the four with no measured counterpart, spread only:")
    print(f"  {'field':<23}{'mean':>6}{'sd':>6}{'mode%':>7}{'distinct':>9}")
    for f in JUDGEMENT_ONLY:
        v = d[f].dropna()
        mo = v.mode().iloc[0]
        print(f"  {f:<23}{v.mean():>6.2f}{v.std():>6.2f}"
              f"{(v == mo).mean()*100:>6.0f}%{v.nunique():>9}")
        rows.append({"field": f, "twin": None, "rho": np.nan, "lo": np.nan,
                     "hi": np.nan, "n": len(v),
                     "mode_pct": (v == mo).mean()*100})

    print(f"\n  do the fields still differ from each other?")
    fam = ["vertical_greenery", "green_eye_level", "green_softening"]
    c = d[fam].corr(method="spearman").values[np.triu_indices(3, 1)]
    print(f"    greenery trio inter-correlation {np.round(c, 2).tolist()}"
          f"   (1.00 under the old schema)")

    print(f"\n  left against right, same node, different sidewalk:")
    for f in ["vertical_greenery", "sky_openness", "walkable_ground",
              "facade_variation"]:
        s = d.groupby("side")[f].mean()
        print(f"    {f:<23}L {s.get('L', np.nan):.2f}   R {s.get('R', np.nan):.2f}")

    out = args.out or (RES / "tables" / "sim_vlm_validation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    d.to_csv(RES / "tables" / "sim_vlm_with_arcs.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
