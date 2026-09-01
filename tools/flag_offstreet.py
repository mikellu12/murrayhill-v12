"""Flag half-views that are not street-level public space.

Three failure modes reach the frame set, and Street View's own source=outdoor
filter -- which s02 already requests -- stops none of them:

  interiors     shop and office lobbies, station concourses
  subways       the pedestrian tunnels under Bank and King William Street
  vehicles      user photospheres shot from a tour boat or an open-top bus

ONE TEST COVERS ALL THREE, because they share what they lack rather than what
they contain: a frame of street-level public space has public GROUND in it.
Road, sidewalk, pedestrian area or kerb, from either segmenter. An interior's
floor segments as floor or as nothing; a subway reads as tunnel; and a
photograph taken from a boat has no pavement in it at any distance.

    off_street = ground < 0.05  OR  map_Tunnel > 0.05

MEASURED ON 56 HAND-LABELLED FRAMES: precision 0.94, recall 0.79. Two of 32
flagged frames were real streets; 8 of 38 bad frames were missed.

WHAT THIS REPLACES, AND WHY THE FIRST ATTEMPT LOOKED BETTER THAN IT WAS. The
first rule keyed on an absent sky, an absent road and a present ade_ceiling. It
scored 0.70/0.70 on a first batch of 16 -- but that batch was drawn FROM its
own flagged set, two flagged and one kept per street, which cannot measure
recall. Scored against a stratified batch it catches 29% of bad frames. A
detector evaluated on its own positives will always look calibrated.

THE CEILING TEST WAS ALSO WRONG IN KIND. ade_ceiling fires on the glazed roof
of Leadenhall Market, which is a public pedestrian street with shopfronts on
both sides -- exactly what the SIM exists to measure -- and fails to fire in
interiors where the ceiling is out of frame. Roofed thoroughfares count as
street interface; the ground test keeps them, because they are paved.

The flag is written, not applied. Whether an interior belongs in a street
measure is the analyst's call, and a silent row drop hides it.

    .venv/Scripts/python tools/flag_offstreet.py
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

GROUND_MIN, TUNNEL_MAX = 0.05, 0.05
GROUND = ["map_Road", "map_Sidewalk", "map_Pedestrian Area", "map_Curb",
          "ade_sidewalk", "ade_road"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    banner("flag frames that are not street-level public space")

    seg = pd.read_csv(args.seg or PROC / "seg90_two_model.csv")
    col = lambda c: seg[c] if c in seg else pd.Series(0.0, index=seg.index)
    ground = sum(col(c) for c in GROUND)
    tunnel = col("map_Tunnel")

    d = pd.DataFrame({"file": seg.file, "ground_share": ground,
                      "tunnel_share": tunnel})
    d["off_street"] = (ground < GROUND_MIN) | (tunnel > TUNNEL_MAX)
    d["reason"] = np.where(tunnel > TUNNEL_MAX, "tunnel",
                    np.where(ground < GROUND_MIN, "no public ground", ""))
    d["node_id"] = [re.search(r"(n\d+)", f).group(1) for f in d.file]

    out = args.out or PROC / "offstreet_flag.csv"
    d.to_csv(out, index=False)
    n = int(d.off_street.sum())
    print(f"{len(d)} frames, {n} flagged ({n/len(d)*100:.2f}%), "
          f"{d[d.off_street].node_id.nunique()} nodes")
    print("  " + d[d.off_street].reason.value_counts().to_string().replace("\n", "\n  "))

    lp = PROC / "frame_labels.csv"
    if lp.exists():
        L = pd.read_csv(lp).merge(d[["file", "off_street"]], on="file", how="left")
        bad = L.label != "street"
        tp = int((L.off_street & bad).sum()); fp = int((L.off_street & ~bad).sum())
        fn = int((~L.off_street & bad).sum())
        print(f"\n  against {len(L)} hand-labelled frames: "
              f"precision {tp/max(tp+fp,1):.2f}, recall {tp/max(tp+fn,1):.2f}")

    calc = RES / "tables" / "vlm_calculations.csv"
    if calc.exists():
        c = pd.read_csv(calc).merge(d[["file", "off_street"]], on="file", how="left")
        c["off_street"] = c.off_street.fillna(False)
        print(f"\n  {'':<18}{'n':>7}{'median M':>10}{'mean':>8}")
        for lab, m in (("all", pd.Series(True, index=c.index)),
                       ("flagged", c.off_street), ("kept", ~c.off_street)):
            print(f"  {lab:<18}{int(m.sum()):>7}{c.loc[m,'M'].median():>10.3f}"
                  f"{c.loc[m,'M'].mean():>8.3f}")
        g = c.groupby("street").agg(n=("M", "size"), flagged=("off_street", "sum"),
                                    M_all=("M", "median"))
        g["M_kept"] = c[~c.off_street].groupby("street")["M"].median()
        g = g[g.flagged > 0]
        g["dM"] = g.M_kept - g.M_all
        print(f"\n  streets moving most when the flagged frames are dropped:")
        print(f"    {'street':<30}{'flagged':>10}{'M all':>8}{'M kept':>9}{'shift':>8}")
        for s, r in g.reindex(g.dM.abs().sort_values(ascending=False).index).head(8).iterrows():
            k = f"{r.M_kept:.3f}" if pd.notna(r.M_kept) else "  --"
            print(f"    {str(s)[:29]:<30}{int(r.flagged):>4}/{int(r.n):<5}"
                  f"{r.M_all:>8.3f}{k:>9}{r.dM:>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
