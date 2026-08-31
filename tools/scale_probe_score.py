"""Four cells -- two anchors or seven, argmax or expected value -- scored.

scale_probe.py records all four for the same images in the same pass, so
nothing differs but the prompt and the read. Each is scored against the share
measured over that half-view's own 90 degrees.

    .venv/Scripts/python tools/scale_probe_score.py
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
        "sky_openness": (1, "sky share")}
CELLS = [("a2", "argmax"), ("a2", "ev"), ("a7", "argmax"), ("a7", "ev")]
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
    d = pd.read_csv(RES / "tables" / "scale_probe.csv")
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

    print(f"  {'field':<20}{'prompt':<10}{'read':<9}{'rho':>8}   "
          f"{'95% CI':<20}{'distinct':>9}")
    for f, (row, label) in TWIN.items():
        tgt = cols[row]
        for tag, how in CELLS:
            v = d[f"{f}__{tag}_{how}"]
            got = boot(v.to_numpy(float), d[tgt].to_numpy(float), g)
            if not got:
                continue
            r, lo, hi, n = got
            name = "2-anchor" if tag == "a2" else "7-anchor"
            print(f"  {f:<20}{name:<10}{how:<9}{r:>+8.3f}   "
                  f"[{lo:+.3f},{hi:+.3f}]{v.nunique():>9}")
        print()
    print("  facade_variation and walkable_ground have no measured twin;")
    print("  scale_probe.py reports their spread instead.")


if __name__ == "__main__":
    main()
