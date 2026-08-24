"""Street Interface Matrix and the demonstration dwell index.

Reads sim_profiles.npz, slices the 180-degree along-street window at every
node, and assembles the three dimensions of the paper's diagram from pixel
shares that ADE20K can actually deliver.

Two deliberate departures from the LaTeX in the manuscript, both forced by
what the segmenter returns:

1. P_sidewalk / P_paver is not computable. ADE20K's class is "sidewalk,
   pavement" -- paver is not a separate label, so the ratio divides a class
   by itself. It becomes sidewalk / (sidewalk + road): the same question,
   asked of a denominator that cannot be zero in a street view.

2. Every term is a bounded share rather than a ratio of two free quantities.
   H/W runs 0.05 to 10.07 here while SVF is [0,1]; summing them under weights
   that total 1 lets whichever is largest decide the answer. Enclosure enters
   through VEI, which is measured from the same pixels and already bounded.

Enclosure is deliberately non-monotone. Phi(VEI) = 4*VEI*(1-VEI) peaks at
VEI = 0.5, which encodes the manuscript's own claim -- some enclosure defines
the room, too much oppresses -- instead of asserting that more is uniformly
better or worse.

    .venv/Scripts/python tools/sim_dwell.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, bin_mask

FOV = CFG["directional"]["fov"]
W = CFG["sim"]["weights"]
SATQ = CFG["sim"]["saturation_quantile"]

# Terms whose raw share is too sparse to weigh against building or road.
SPARSE = ["green_eye", "articulation", "barrier", "affordance"]


def shares(prof, rows, centre):
    """Class shares of the along-street window, normalised by the weight row."""
    mk = bin_mask(centre, FOV)
    w = prof[rows["weight"]][mk].sum()
    if w <= 0:
        return None
    return {g: prof[i][mk].sum() / w for g, i in rows.items() if g != "weight"}


def main():
    banner("SIM terms and demonstration dwell index")
    z = np.load(PROC / "sim_profiles.npz", allow_pickle=True)
    names = [str(r) for r in z["__rows__"]]
    rows = {r: i for i, r in enumerate(names)}
    m = pd.read_csv(PROC / "metrics.csv")
    # Park Avenue's second carriageway, its tunnel roadway, and nodes past
    # the boundary streets are in the frame but outside the study area.
    import geopandas as gpd
    _n = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in _n.columns:
        keep = set(_n.loc[_n.in_study, "node_id"])
        before = len(m); m = m[m.node_id.isin(keep)]
        print(f"study-area filter: {before} -> {len(m)} nodes")
    axis = dict(zip(m.node_id, m.street_axis_deg))

    recs = []
    for nid in (k for k in z.files if k != "__rows__" and k in axis):
        c = axis.get(nid)
        if c is None or np.isnan(c):
            continue
        s = shares(z[nid], rows, c)
        if s is None:
            continue
        ground = s["eye_green"] + s["road"] + s["sidewalk"]
        walk = s["sidewalk"] + s["road"]
        vei_d = s["building"] + s["sky"]
        recs.append({
            "node_id":      nid,
            "green_eye":    s["eye_green"],
            "green_canopy": s["canopy_green"],
            "green_ground": s["eye_green"] / ground if ground > 0 else 0.0,
            "articulation": s["articulation"],
            "barrier":      s["hard_barrier"],
            "walkable":     s["sidewalk"] / walk if walk > 0 else 0.0,
            # The single sparse affordance term: seating, shelter and soft
            # buffers pooled. Individually they are zero at 68 / 53 / 78 per
            # cent of nodes; pooled, at 33 per cent. Still sparse, but a
            # presence signal rather than three mostly-empty columns.
            "affordance":   s["rest"] + s["shelter"] + s["soft_buffer"],
            "VEI":          s["building"] / vei_d if vei_d > 0 else np.nan,
        })
    d = pd.DataFrame(recs).dropna(subset=["VEI"])
    print(f"nodes with an along-street window: {len(d)}\n")

    # Saturating transform on the sparse terms.
    print(f"saturation scale s0 = q{SATQ:.2f} of each term:")
    for c in SPARSE:
        s0 = d[c].quantile(SATQ)
        s0 = s0 if s0 > 0 else d.loc[d[c] > 0, c].median()
        d[c + "_n"] = 1 - np.exp(-d[c] / s0)
        print(f"  {c:<13} s0={s0:.5f}   median raw={d[c].median():.5f} "
              f"-> normalised={d[c + '_n'].median():.3f}")

    # Some enclosure defines the room; too much oppresses. Peaks at VEI=0.5.
    d["enclosure"] = 4 * d.VEI * (1 - d.VEI)
    d["openness"] = 1 - d.barrier_n

    g, mo, p = W["green"], W["morphological"], W["permeability"]
    d["G"] = g["eye"] * d.green_eye_n + g["ground"] * d.green_ground
    d["M"] = (mo["articulation"] * d.articulation_n
              + mo["enclosure"] * d.enclosure
              + mo["openness"] * d.openness)
    d["P"] = p["walkable"] * d.walkable + p["affordance"] * d.affordance_n
    dw = W["dimension"]
    d["SIM"] = dw["green"] * d.G + dw["morphological"] * d.M + dw["permeability"] * d.P

    print("\n=== dimensions ===")
    for c in ("G", "M", "P", "SIM"):
        print(f"  {c:<4} min {d[c].min():.3f}  median {d[c].median():.3f}  "
              f"max {d[c].max():.3f}")

    out = PROC / "sim_index.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(d)} nodes, {d.shape[1]} columns)")


if __name__ == "__main__":
    main()
