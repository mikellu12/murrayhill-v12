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
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, weights
import sim_core as S
from sim_readout import prune_once, interpolated_median, K

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=None,
                    help="ratings table; default is this frame's own")
    ap.add_argument("--rounded-ev", action="store_true",
                    help="use the stored round(EV) column instead of the "
                         "pruned interpolated median, for comparison")
    args = ap.parse_args()
    banner("SIM per half-view, Cobb-Douglas, VLM inputs")
    C = CFG["sim_vlm"]["cobb_douglas"]
    W = CFG["sim_vlm"]["weights"]
    a_w, b_w, g_w = (weights(W["imageability"]), weights(W["identity"]),
                     weights(W["dependence"]))
    kI, tI = C["imageability_sigmoid"]["kappa"], C["imageability_sigmoid"]["tau"]
    kD, tD = C["dependence_sigmoid"]["kappa"], C["dependence_sigmoid"]["tau"]

    tbl = args.table
    if tbl is None:
        # sim_vlm.csv is the original 2-anchor run, kept for comparison. The
        # current Murray Hill ratings live in sim_vlm_v3.csv.
        tbl = RES / "tables" / "sim_vlm_v3.csv"
        if not tbl.exists():
            tbl = RES / "tables" / "sim_vlm.csv"
    print(f"ratings: {tbl}")
    d = pd.read_csv(tbl)

    # NODES TAGGED usable: False ARE OUT, before anything is computed. The tag
    # lives on nodes.csv (tools/node_usability.py): tunnel interiors and the
    # viaduct deck for Murray Hill, user-contributed panoramas for London --
    # tourist buses and interiors wearing a street's coordinates. Rendering
    # already skips most of these, but a table built earlier, or a study area
    # whose renders predate the tag, can still carry their rows, and a
    # calibration that includes a bus interior moves every threshold.
    dropped_raw = None
    npath = PROC / "nodes.csv"
    if npath.exists():
        nu = pd.read_csv(npath)
        if "usable" in nu.columns:
            bad = set(nu.loc[~nu.usable.astype(bool), "node_id"])
            if "node_id" not in d.columns:
                d["node_id"] = d.file.astype(str).str.extract(r"(n\d+)")[0]
            before = len(d)
            # kept aside, not discarded: the ratings of an unusable frame are
            # still observations and belong in vlm_observations with the tag,
            # they just must not touch the calibration or the calculations
            dropped_raw = d[d.node_id.isin(bad)].copy()
            d = d[~d.node_id.isin(bad)].copy()
            if before - len(d):
                print(f"set aside {before - len(d)} rows on "
                      f"{len(bad)} unusable nodes (tagged in "
                      f"vlm_observations, absent from vlm_calculations)")

    # A study area without building footprints has no H/W, and therefore no
    # metrics.csv. Those columns are filled with NaN rather than skipped, so
    # the code below is one path: regime_exponents already falls back to the
    # global elasticities wherever H/W is NaN, and A_i is handled at the
    # calibration loop.
    mpath = PROC / "metrics.csv"
    if mpath.exists():
        hw = pd.read_csv(mpath)[["node_id", "HW_facade", "HW_effective",
                                 "HW_source", "face_id"]]
        d = d.merge(hw, on="node_id", how="left")
    else:
        print(f"no {mpath.name}: no H/W in this frame, so A_i = 1 for every "
              f"node and the global elasticities apply throughout")
        for c, v in (("HW_facade", np.nan), ("HW_effective", np.nan),
                     ("HW_source", None), ("face_id", None)):
            d[c] = v

    # READOUT. The bare field column in the ratings table is round(EV), and
    # expected value is not a defensible summary of an ordinal scale: it
    # asserts that the step from rung 2 to 3 equals the step from 6 to 7, and
    # it turns a frame split between 2 and 6 into a 4, a rung the model
    # positively rejected. The seven-rung distribution is saved for exactly
    # this reason, so M is built from the pruned interpolated median instead --
    # a quantile, which needs the order of the rungs and never their spacing.
    #
    # It changes the ranking very little (Spearman 0.966 against round(EV) on
    # Murray Hill) and does not buy separability at M, where per-field gains
    # cancel through the Cobb-Douglas. The reason to prefer it is that it is
    # the right statistic for the scale, not that it scores better.
    have_p = all(f"{f}_p1" in d.columns for f in TERMS.values())
    if have_p and not args.rounded_ev:
        print("readout: prune one rung, then the interpolated median")
        def rung(f):
            P = d[[f"{f}_p{k}" for k in K]].to_numpy(float)
            P = P / P.sum(axis=1, keepdims=True)
            return pd.Series(interpolated_median(prune_once(P)), index=d.index)
    else:
        print("readout: round(EV) as stored" if have_p else
              "readout: round(EV) as stored (no per-rung columns in this table)")
        def rung(f):
            return d[f].astype(float)

    for term, field in TERMS.items():
        d[term] = (rung(field).clip(1, 7) - 1) / 6.0

    # V_nat / V_built is a ratio of two ratings, so it is built from the raw
    # values rather than the normalised ones -- r/(1+r) form, bounded, and it
    # cannot reach 0 or 1 because neither rating can be 0.
    vg = rung("vertical_greenery").clip(1, 7)
    vh = rung("vertical_hardscape").clip(1, 7)
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
    # A frame with no H/W anywhere has no median to centre A_i on. nanmedian
    # of an all-NaN column is NaN and would propagate into every M, so it falls
    # back to the stated threshold; A_i is 1 throughout in that case anyway.
    hw_local = (float(np.nanmedian(d.HW_effective))
                if d.HW_effective.notna().any() else C["omega"]["hw_threshold"])
    CAL = {"local": dict(tI=float(d.I_raw.median()), tD=float(d.D_raw.median()),
                         hw=hw_local),
           "global": dict(tI=tI, tD=tD, hw=C["omega"]["hw_threshold"])}
    for tag, c in CAL.items():
        suf = "" if tag == "global" else "_local"
        I = S.sigmoid(d.I_raw, kI, c["tI"])
        D = S.sigmoid(d.D_raw, kD, c["tD"])
        # A_i discounts on H/W, a plan geometry. Where the frame has no
        # heights at all there is nothing to discount on, so A_i is 1 for every
        # node -- not NaN, which would void every M, and not a guess at the
        # ratio. It makes M an un-penalised I^a Y^b D^c, which is the same
        # quantity the M_noA column reports for the frames that do have
        # heights, so the two remain comparable.
        if d.HW_effective.notna().any():
            om = S.omega(d.HW_effective, C["omega"]["psi"], c["hw"],
                         open_one_side=porous)
        else:
            om = pd.Series(1.0, index=d.index)
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

    mpath = PROC / "metrics.csv"
    j = d
    if mpath.exists():
        print(f"\n  M by typology:")
        tp = pd.read_csv(mpath)[["node_id", "typology"]]
        j = d.merge(tp, on="node_id", how="left")
        print(j.groupby("typology")["M"].agg(["count", "mean", "std"]).round(3).to_string())
        print("\n  paper reports: mid-block 0.78, avenue canyon 0.22, POPS 0.65")
    elif (PROC / "street_type.csv").exists():
        # No typology in this frame. Street type is the split that exists, and
        # it is the one that governs how the numbers may be read: a 180-degree
        # pedestrian strip and a 90-degree vehicular half are different fields
        # of view, so their Ms are not strictly on one scale and belong in
        # separate rows rather than a pooled mean.
        t = pd.read_csv(PROC / "street_type.csv")[["node_id", "is_pedestrian"]]
        j = d.merge(t, on="node_id", how="left")
        j["view"] = np.where(j.is_pedestrian.fillna(False),
                             "pedestrian 180", "vehicular 90")
        print(f"\n  M by street type, reported separately because the field "
              f"of view differs:")
        print(j.groupby("view")["M"].agg(["count", "mean", "std"]).round(3).to_string())

    print(f"\n  left against right, same node:")
    print(j.groupby("side")[["I", "Y", "D", "M"]].mean().round(3).to_string())

    # The two deliverables are vlm_observations.csv and vlm_calculations.csv.
    # There is no intermediate table: sim_export does the observed/derived
    # split on the frame in memory, so the names in the results folder are
    # the names the study uses.
    import sim_export
    sim_export.write_split(d, dropped=dropped_raw)


if __name__ == "__main__":
    main()
