"""M_i for every half-view, from the VLM ratings, per manuscript section 2.7.

Every term in I, Y and D comes from the VLM. That is the point of the study --
how the model sees the street -- so the pixel measurements are not inputs here.
They stay a separate arm for comparison against segmentation masking later.

Omega is the one exception and it is deliberate: it discounts on H/W, a plan
geometry measured from footprints and facade width in s05, and a single
eye-level view cannot see the street width in the denominator. Where H/W is
missing, Omega is NaN and M is NaN with it -- substituting 1.0 would hand every
unmeasured node full marks. The one exception is a node s05 flagged
open_one_side: there the missingness is not a failed measurement but a street
with no second wall, which is the manuscript's porous block edge. Those take
Omega = 1 and the POPS elasticities, because section 2.8 gives that regime no
H/W band and it is otherwise unreachable.

    .venv/Scripts/python tools/sim_compute.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, weights
import sim_core as S

# manuscript term -> the VLM field that supplies it
TERMS = {
    "GVI_eye": "green_eye_level",
    "GMI": "green_softening",
    "V_sign": "signage_detail",
    "SFV": "facade_variation",
    "V_pave": "walkable_ground",
    "GFAPI": "ground_floor_activity",
    "SVF": "sky_openness",
    "IAS": "resting_affordance",
}


def main():
    banner("SIM per half-view, Cobb-Douglas, VLM inputs")
    C = CFG["sim_vlm"]["cobb_douglas"]
    W = CFG["sim_vlm"]["weights"]
    a_w, b_w, g_w = (weights(W["imageability"]), weights(W["identity"]),
                     weights(W["dependence"]))
    kI, tI = C["imageability_sigmoid"]["kappa"], C["imageability_sigmoid"]["tau"]
    kD, tD = C["dependence_sigmoid"]["kappa"], C["dependence_sigmoid"]["tau"]

    # sim_vlm.csv is the original 2-anchor run and is kept for comparison.
    # The current ratings live in sim_vlm_v3.csv: same nine fields, corrected
    # rung sets, and all 766 nodes rather than the 681 that had imagery then.
    tbl = RES / "tables" / "sim_vlm_v3.csv"
    if not tbl.exists():
        tbl = RES / "tables" / "sim_vlm.csv"
    print(f"ratings: {tbl.name}")
    d = pd.read_csv(tbl)
    hw = pd.read_csv(PROC / "metrics.csv")[["node_id", "HW_facade",
                                            "HW_effective", "HW_source", "face_id"]]
    d = d.merge(hw, on="node_id", how="left")

    # 1-7 -> [0,1]
    n = lambda c: (d[c].clip(1, 7) - 1) / 6.0
    for term, field in TERMS.items():
        d[term] = n(field)

    # V_nat / V_built is a ratio of two ratings, so it is built from the raw
    # values rather than the normalised ones -- r/(1+r) form, bounded, and it
    # cannot reach 0 or 1 because neither rating can be 0.
    vg = d.vertical_greenery.clip(1, 7)
    vh = d.vertical_hardscape.clip(1, 7)
    d["nat_built"] = vg / (vg + vh)

    d["I_raw"] = (a_w["nat_built"] * d.nat_built + a_w["gvi_eye"] * d.GVI_eye
                  + a_w["gmi"] * d.GMI)
    d["Y"] = (b_w["signboard"] * d.V_sign + b_w["enclosure"] * (1 - d.SVF)
              + b_w["sfv"] * d.SFV)
    d["D_raw"] = (g_w["sidewalk_paver"] * d.V_pave + g_w["ias"] * d.IAS
                  + g_w["gfapi"] * d.GFAPI)
    porous = (d.HW_source.eq("open_one_side") if "HW_source" in d.columns
              else None)

    # ---- two calibrations, because one number cannot answer both questions --
    # LOCAL is CWMC applied as the protocol states it (Nature.8.31 para 129):
    # "dynamically set to the city-wide median values of raw metrics". It
    # centres the sigmoid on THIS frame, so M is a relative index within the
    # study area and the ranking has full variance -- but every city's median
    # lands at 0.5 by construction, so two cities cannot be compared on it.
    #
    # GLOBAL uses the thresholds the manuscript states, which carry an external
    # justification the median does not: tau_I = 0.20 is defended as the
    # minimum foveal vegetation for cognitive restoration, separating canyon
    # GVI_eye <= 0.06 from mid-block 0.22. Fixed across cities, so London and
    # Murray Hill sit on one scale -- at the cost of saturating wherever a
    # city's distribution sits far from it.
    #
    # For this frame they agree on D (median D_raw 0.500 against the stated
    # 0.50) and differ on I (0.395 against 0.20). Ranking is barely affected:
    # rho +0.930 between the two Ms.
    CAL = {"local": dict(tI=float(d.I_raw.median()), tD=float(d.D_raw.median()),
                         hw=float(np.nanmedian(d.HW_effective))),
           "global": dict(tI=tI, tD=tD, hw=C["omega"]["hw_threshold"])}
    for tag, c in CAL.items():
        suf = "" if tag == "global" else "_local"
        I = S.sigmoid(d.I_raw, kI, c["tI"])
        D = S.sigmoid(d.D_raw, kD, c["tD"])
        om = S.omega(d.HW_effective, C["omega"]["psi"], c["hw"],
                     open_one_side=porous)
        e = S.regime_exponents(d.HW_effective.to_numpy(), C, porous=porous)
        d["I" + suf], d["D" + suf], d["Omega" + suf] = I, D, om
        d["a" + suf], d["b" + suf], d["c" + suf] = (
            e["imageability"], e["identity"], e["dependence"])
        d["M" + suf] = S.matrix_score(I, d.Y, D, om, e["imageability"],
                                      e["identity"], e["dependence"])
        # ...and the same score with the canyon penalty switched off. A study
        # area without building heights cannot compute A_i at all, so its M is
        # I^a Y^b D^c with nothing discounting it and sits systematically
        # higher -- Murray Hill's median moves 0.472 to 0.610 when Omega is
        # dropped, and the minimum rises 0.016 to 0.271, because the penalty is
        # what creates the bottom of the scale. Comparing a city that has it
        # against one that does not compares the presence of the term, not the
        # streets. This column is the like-for-like one.
        d["M" + suf + "_noA"] = S.matrix_score(
            I, d.Y, D, 1.0, e["imageability"], e["identity"], e["dependence"])
    for tag, c in CAL.items():
        print(f"  tau {tag:<7} I {c['tI']:.3f}   D {c['tD']:.3f}   "
              f"Omega_th {c['hw']:.3f}")
    print()

    print(f"{len(d)} half-views, {d.node_id.nunique()} nodes")
    print(f"{d.M.notna().sum()} with a score, "
          f"{d.M.isna().sum()} lost to a missing H/W\n")
    print(f"  {'':<12}{'min':>8}{'median':>9}{'mean':>8}{'max':>8}{'sd':>8}")
    for c in ["I_raw", "I", "I_local", "Y", "D_raw", "D", "D_local",
              "Omega", "Omega_local", "M", "M_local",
              "M_noA", "M_local_noA"]:
        v = d[c].dropna()
        print(f"  {c:<12}{v.min():>8.3f}{v.median():>9.3f}{v.mean():>8.3f}"
              f"{v.max():>8.3f}{v.std():>8.3f}")

    sat_I = float((d.I > 0.95).mean() + (d.I < 0.05).mean())
    sat_D = float((d.D > 0.95).mean() + (d.D < 0.05).mean())
    print(f"\n  sigmoid saturation (below 0.05 or above 0.95):")
    print(f"    I  {sat_I*100:>5.1f}%      D  {sat_D*100:>5.1f}%")
    if sat_D > 0.5 or sat_I > 0.5:
        print("    A dimension pinned at its rail carries no information into M.")
        print("    tau_I and tau_D were calibrated against pixel shares; the")
        print("    normalised 1-7 ratings do not sit on that scale.")

    print(f"\n  M by typology:")
    tp = pd.read_csv(PROC / "metrics.csv")[["node_id", "typology"]]
    j = d.merge(tp, on="node_id", how="left")
    print(j.groupby("typology")["M"].agg(["count", "mean", "std"]).round(3).to_string())
    print("\n  paper reports: mid-block 0.78, avenue canyon 0.22, POPS 0.65")

    print(f"\n  left against right, same node:")
    print(j.groupby("side")[["I", "Y", "D", "M"]].mean().round(3).to_string())

    # The two deliverables are vlm_observations.csv and vlm_calculations.csv.
    # There is no intermediate table: sim_export does the observed/derived
    # split on the frame in memory, so the names in the results folder are
    # the names the study uses.
    import sim_export
    sim_export.write_split(d)


if __name__ == "__main__":
    main()
