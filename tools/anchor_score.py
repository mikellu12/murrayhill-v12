"""Score anchor_probe.csv against the measured share over each 90-degree arc.

Split out of anchor_probe.py because the bearings come from nodes.gpkg, and
geopandas lives in .venv, not .venv-gpu. That split is deliberate -- the GPU
env is torch and transformers only -- so anything needing the frame runs here
afterwards rather than being imported into a model script.

    .venv/Scripts/python tools/anchor_score.py
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

# only vertical_hardscape has a measured counterpart; the other two are judged
# on spread alone, which anchor_probe.py already prints
TWIN = {"vertical_hardscape": 2}  # profile row 2 is the building share


def main():
    banner("anchor variants against the measured arc")
    d = pd.read_csv(RES / "tables" / "anchor_probe.csv")
    z = np.load(PROC / "azimuth_profiles.npz")
    prof = {k: z[k] for k in z.files}
    bear = walk_bearings()

    shares = {name: [] for name in ("veg", "sky", "bld")}
    for r in d.itertuples():
        p, b = prof.get(r.node_id), bear.get((r.street, r.walk))
        if p is None or b is None or r.side not in SIDE_OFF:
            for k in shares:
                shares[k].append(np.nan)
            continue
        m = bin_mask((b + SIDE_OFF[r.side]) % 360, 90.0)
        W = p[3][m].sum()
        for i, k in enumerate(("veg", "sky", "bld")):
            shares[k].append(p[i][m].sum() / W if W > 0 else np.nan)
    for k, v in shares.items():
        d["arc_" + k] = v

    n = d.arc_bld.notna().sum()
    print(f"  {n} of {len(d)} rows matched to a profile arc")
    print(f"  building share over the arc: {d.arc_bld.min():.3f} to "
          f"{d.arc_bld.max():.3f}\n")
    print(f"  {'field':<20}{'variant':<9}{'rho vs measured':>17}{'mode%':>8}")
    for field, row in TWIN.items():
        tgt = {2: "arc_bld"}[row]
        for v in ("a", "b"):
            col = f"{field}__{v}"
            if col not in d:
                continue
            s = d[[col, tgt]].dropna()
            if len(s) < 8 or s[col].nunique() < 2:
                print(f"  {field:<20}{v:<9}{'too few distinct':>17}")
                continue
            rho = s[col].corr(s[tgt], method="spearman")
            vals = d[col].dropna()
            mo = vals.mode().iloc[0]
            print(f"  {field:<20}{v:<9}{rho:>+17.3f}"
                  f"{(vals == mo).mean()*100:>7.0f}%")
    print("\n  walkable_ground and facade_variation have no measured "
          "counterpart;\n  spread is the only test available for them.")


if __name__ == "__main__":
    main()
