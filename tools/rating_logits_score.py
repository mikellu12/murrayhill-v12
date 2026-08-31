"""Does the expected value track the street better than the argmax?

rating_logits.py records both for the same forward pass, so this is a paired
comparison on identical inputs -- no sampling difference, no second run. Each
is scored against the share measured over that half-view's own 90 degrees.

    .venv/Scripts/python tools/rating_logits_score.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner, bin_mask
from half_target import walk_bearings, SIDE_OFF

TWIN = {"green_eye_level": (0, "vegetation share"),
        "vertical_greenery": (0, "vegetation share"),
        "sky_openness": (1, "sky share"),
        "facade_variation": (None, "no measured counterpart")}
RNG = np.random.default_rng(0)


def boot(x, y, g, n=3000):
    s = pd.DataFrame({"x": x, "y": y, "g": g}).dropna()
    if len(s) < 20 or s.x.nunique() < 2:
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
    banner("expected value against argmax")
    d = pd.read_csv(RES / "tables" / "rating_logits.csv")
    z = np.load(PROC / "azimuth_profiles.npz")
    prof = {k: z[k] for k in z.files}
    bear = walk_bearings()
    parts = d.file.str.split("/", expand=True)
    d["dir_street"], d["dir_walk"] = parts[0], parts[1]

    cols = {0: "arc_veg", 1: "arc_sky", 2: "arc_bld"}
    acc = {v: [] for v in cols.values()}
    for r in d.itertuples():
        p, b = prof.get(r.node_id), bear.get((r.dir_street, r.dir_walk))
        if p is None or b is None or r.side not in SIDE_OFF:
            for v in acc.values():
                v.append(np.nan)
            continue
        m = bin_mask((b + SIDE_OFF[r.side]) % 360, 90.0)
        W = p[3][m].sum()
        for i, name in cols.items():
            acc[name].append(p[i][m].sum() / W if W > 0 else np.nan)
    for k, v in acc.items():
        d[k] = v
    face = pd.read_csv(PROC / "metrics.csv")[["node_id", "face_id"]]
    d = d.merge(face, on="node_id", how="left")
    g = d.face_id.fillna(d.node_id).to_numpy()
    print(f"{len(d)} half-views, {d.face_id.nunique()} faces\n")

    print(f"  {'field':<20}{'read as':<9}{'rho':>8}   {'95% CI':<20}"
          f"{'distinct':>9}")
    for f, (row, label) in TWIN.items():
        if row is None:
            for how in ("argmax", "ev"):
                v = d[f"{f}_{how}"]
                print(f"  {f:<20}{how:<9}{'--':>8}   {'no twin':<20}"
                      f"{v.nunique():>9}")
            print()
            continue
        tgt = cols[row]
        for how in ("argmax", "ev"):
            v = d[f"{f}_{how}"]
            got = boot(v.to_numpy(float), d[tgt].to_numpy(float), g)
            if not got:
                continue
            r, lo, hi, n = got
            print(f"  {f:<20}{how:<9}{r:>+8.3f}   [{lo:+.3f},{hi:+.3f}]"
                  f"{v.nunique():>9}")
        print()

    print("  what the argmax throws away:")
    for f in TWIN:
        top = d[f"{f}_top"].mean()
        print(f"    {f:<20}the chosen digit carries {top*100:>4.0f}% of the mass")


if __name__ == "__main__":
    main()
