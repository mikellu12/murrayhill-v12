"""Flag half-views whose panorama is an interior, not a street.

Street View's outdoor filter is already requested -- s02 sends source=outdoor
on every metadata probe -- and interiors still come through: station subways
under King William Street, shop and office lobbies, the passages under Bank.
Google's own classification is not tight enough to rely on alone, so this is a
second pass over frames already segmented, at no extra cost.

THREE CONDITIONS, ALL REQUIRED, because no one of them is sufficient in the
City of London:

  no sky        below 1% of the frame. On its own this is useless here -- the
                City is a deep canyon and 48% of frames sit below 5% sky, so a
                sky threshold alone would flag half the study area.
  no road       below 2%. Separates an interior from a roofed but trafficked
                street.
  a ceiling     ade_ceiling above 2%. This is the discriminating one: an
                outdoor canyon has no ceiling class anywhere in it.

CONFIRMED BY EYE, not assumed. The flagged frames were rendered and checked;
they are interiors and subway tunnels throughout.

The flag is written, not applied. Dropping rows here would make the exclusion
invisible in the tables, and the choice of whether an interior belongs in a
street measure is the analyst's, not this script's.

    .venv/Scripts/python tools/flag_indoor.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

SKY_MAX, ROAD_MAX, CEIL_MIN = 0.01, 0.02, 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    banner("flag interior and subway frames")

    seg = pd.read_csv(args.seg or PROC / "seg90_two_model.csv")
    sky = seg[[c for c in ("map_Sky", "ade_sky") if c in seg]].max(axis=1)
    road = seg[[c for c in ("map_Road", "ade_road") if c in seg]].max(axis=1)
    ceil = seg["ade_ceiling"] if "ade_ceiling" in seg else pd.Series(0.0, index=seg.index)

    d = pd.DataFrame({"file": seg.file, "sky_share": sky, "road_share": road,
                      "ceiling_share": ceil})
    d["indoor"] = (sky < SKY_MAX) & (road < ROAD_MAX) & (ceil > CEIL_MIN)
    import re
    d["node_id"] = [re.search(r"(n\d+)", f).group(1) for f in d.file]

    out = args.out or PROC / "indoor_flag.csv"
    d.to_csv(out, index=False)
    n = int(d.indoor.sum())
    print(f"{len(d)} frames, {n} flagged ({n/len(d)*100:.2f}%), "
          f"{d[d.indoor].node_id.nunique()} nodes\n")

    calc = RES / "tables" / "vlm_calculations.csv"
    if calc.exists():
        c = pd.read_csv(calc).merge(d[["file", "indoor"]], on="file", how="left")
        c["indoor"] = c.indoor.fillna(False)
        print(f"  {'':<20}{'n':>7}{'median M':>10}{'mean':>8}")
        for lab, m in (("all", pd.Series(True, index=c.index)),
                       ("flagged", c.indoor), ("kept", ~c.indoor)):
            print(f"  {lab:<20}{int(m.sum()):>7}{c.loc[m,'M'].median():>10.3f}"
                  f"{c.loc[m,'M'].mean():>8.3f}")
        # A street that loses half its frames is a street whose number changed,
        # even when the city-wide median barely moves.
        g = c.groupby("street").agg(n=("M", "size"), flagged=("indoor", "sum"),
                                    M_all=("M", "median"))
        g["M_kept"] = c[~c.indoor].groupby("street")["M"].median()
        g["frac"] = g.flagged / g.n
        g = g[g.flagged > 0].sort_values("frac", ascending=False)
        print(f"\n  streets most affected:")
        print(f"    {'street':<30}{'flagged':>9}{'M all':>8}{'M kept':>9}{'shift':>8}")
        for s, r in g.head(10).iterrows():
            sh = r.M_kept - r.M_all if pd.notna(r.M_kept) else np.nan
            print(f"    {str(s)[:29]:<30}{int(r.flagged):>4}/{int(r.n):<4}"
                  f"{r.M_all:>8.3f}{r.M_kept:>9.3f}{sh:>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
