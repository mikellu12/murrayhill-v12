"""Do the model's words agree with the model's numbers?

The walk-through puts a description beside a rating, so a reader will treat
them as one statement. They are not: the ratings are read from logits at a
forced prefix and the text is generated, a different decoding path over the
same image. Nothing makes them consistent, and where they disagree the page
looks broken even when both halves are defensible.

This counts the disagreements instead of arguing about them. For each question
that has a matching field, the text is reduced to a yes/no -- does it say the
thing is present -- and set against the rung the ratings give. Three outcomes
matter:

  CONTRADICTION       the text denies what the rating scores high, or asserts
                      what the rating scores low
  AGREEMENT           both say present, or both say absent
  MEASURED ARBITER    segmentation says which of the two is right, where a
                      pixel share can settle it

THE ARBITER IS THE POINT. Text-versus-rating alone says only that they differ;
it cannot say which to believe. Vegetation has a measured twin, so for the
greenery question the disagreement can be adjudicated rather than reported.
For the others it cannot, and the count is left as a count.

    .venv/Scripts/python tools/describe_vs_score.py
    .venv/Scripts/python tools/describe_vs_score.py \\
        --descriptions results/tables/vlm_descriptions_180.csv \\
        --greenery-override results/tables/vlm_greenery_open.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from sim_readout import prune_once, interpolated_median, K

# Each question, the field whose rung it should track, and the pattern that
# reads the text as a denial. The patterns are deliberately generous to the
# text: anything not clearly a denial counts as an assertion, so a model that
# hedges is credited with saying the thing is there.
DENY = {
    "greenery": re.compile(
        r"\b(no|not|none|without|absent|lacks?)\b[^.]{0,40}"
        r"\b(vegetation|greenery|plants?|trees?|planting)\b"
        r"|\bno visible\b|\bnone\b", re.I),
    "ground": re.compile(
        r"\b(no|not|nothing|none|without)\b[^.]{0,40}"
        r"\b(bench|benches|seat|seating|stool|chairs?)\b", re.I),
    # "a mix of entrances, windows, and blank walls" is not a denial; it is a
    # description that happens to contain the word. Only a frontage described
    # as predominantly closed counts, or the word would fire on almost every
    # honest answer.
    "frontage": re.compile(
        r"\b(no shopfronts?|no entrances?|no windows?|no visible entrances?|"
        r"(mostly|entirely|largely|primarily) (blank|closed|solid|windowless)|"
        r"windowless|boarded up)\b", re.I),
}
# question -> (field, whether a denial means a LOW rung)
# greenery pairs with vertical_greenery, NOT green_eye_level: the arbiter is
# whole-frame map_Vegetation, which is vertical_greenery's twin. Judged against
# green_eye_level the comparison scored a frame with 15% vegetation as a
# contradiction because street-tree canopy sits ABOVE eye level -- the rating
# was right and the arbiter was measuring something else.
PAIRS = {
    "greenery": ("vertical_greenery", True),
    "ground": ("resting_affordance", True),
    "frontage": ("ground_floor_activity", True),
}

# A question the model was never asked to answer cannot contradict anything.
# The ground question asks about the footway, not specifically about seating,
# so most answers simply do not mention it -- and reading silence as "there is
# seating" scored 96 per cent of frames as contradictions, which measured my
# rule and not the model.
MENTIONS = {
    "ground": re.compile(r"(bench|benches|seat|seating|stool|chairs?|"
                         r"planter|bollard|step)", re.I),
}


def readout(d, field):
    cols = [f"{field}_p{k}" for k in K]
    if not all(c in d.columns for c in cols):
        return None
    P = d[cols].to_numpy(float)
    return interpolated_median(prune_once(P / P.sum(axis=1, keepdims=True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=Path,
                    default=RES / "tables" / "sim_vlm_180_placeless.csv")
    ap.add_argument("--descriptions", type=Path,
                    default=RES / "tables" / "vlm_descriptions_180.csv")
    ap.add_argument("--greenery-override", type=Path, default=None,
                    help="a re-run of one question, merged over the original; "
                         "use it to measure whether a prompt change helped")
    ap.add_argument("--seg", type=Path, default=PROC / "seg90_two_model.csv")
    ap.add_argument("--hi", type=float, default=4.0, help="rung counted as present")
    ap.add_argument("--lo", type=float, default=2.5, help="rung counted as absent")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    banner("do the words agree with the numbers?")

    r = pd.read_csv(args.ratings)
    d = pd.read_csv(args.descriptions)
    if args.greenery_override and args.greenery_override.exists():
        o = pd.read_csv(args.greenery_override)[["file", "greenery"]]
        d = d.drop(columns=["greenery"]).merge(o, on="file", how="left")
        print(f"greenery replaced from {args.greenery_override.name}")

    j = r.merge(d, on="file", how="inner", suffixes=("", "_desc"))
    print(f"{len(j)} frames with both a rating and a description\n")

    # segmentation can only arbitrate vegetation, and only per 90 half, so the
    # two halves of a node are averaged back to the strip the model saw
    seg = pd.read_csv(args.seg)
    seg["node_id"] = seg.file.astype(str).str.extract(r"(n\d+)")[0]
    veg = seg.groupby("node_id").map_Vegetation.mean().rename("veg")
    if "node_id" not in j.columns:
        j["node_id"] = j.file.astype(str).str.extract(r"(n\d+)")[0]
    j = j.merge(veg, on="node_id", how="left")

    print(f"{'question':<12}{'field':<24}{'n':>6}{'agree':>8}{'contra':>8}"
          f"{'text says absent':>18}")
    rows = []
    for q, (field, _) in PAIRS.items():
        if q not in j.columns:
            continue
        read = readout(j, field)
        if read is None:
            continue
        txt = j[q].astype(str)
        says_absent = txt.apply(lambda t: bool(DENY[q].search(t)))
        # only frames where the text actually addresses the thing
        if q in MENTIONS:
            spoke = txt.apply(lambda t: bool(MENTIONS[q].search(t)))
            says_absent = says_absent & spoke
        else:
            spoke = pd.Series(True, index=j.index)
        # ONLY DENIALS CAN CONTRADICT. The text is binary and the rung is
        # graded, so "street trees are visible" beside a rung of 2 is not a
        # conflict -- a few trees are honestly both. Counting those scored 38
        # per cent of frames as contradictions and measured the threshold, not
        # the model. A denial is different: "no vegetation" beside a high rung
        # means one of the two is wrong, and segmentation can say which.
        rung_hi, rung_lo = read >= args.hi, read <= args.lo
        contra = spoke & says_absent & rung_hi
        agree = spoke & ((says_absent & rung_lo)
                         | (~says_absent & rung_hi))
        n = int(spoke.sum())
        print(f"{q:<12}{field:<24}{n:>6}{agree.mean():>7.0%}"
              f"{contra.mean():>8.0%}{says_absent.mean():>17.0%}")
        rows.append(dict(question=q, field=field, n=n,
                         agree=float(agree.mean()),
                         contradiction=float(contra.mean()),
                         text_absent=float(says_absent.mean())))
        j[q + "_absent"] = says_absent
        j[q + "_rung"] = read
        j[q + "_contra"] = contra

    pd.DataFrame(rows).to_csv(RES / "tables" / "describe_vs_score.csv",
                              index=False)

    # only greenery can be adjudicated: who is right when they disagree
    if "greenery_contra" in j.columns and j.veg.notna().any():
        c = j[j.greenery_contra & j.veg.notna()]
        if len(c):
            print(f"\nwhere the greenery text and rung disagree ({len(c)} "
                  f"frames), segmentation says:")
            said_absent = c[c.greenery_absent]
            said_present = c[~c.greenery_absent]
            if len(said_absent):
                print(f"  text said ABSENT, rung high: measured vegetation "
                      f"median {said_absent.veg.median()*100:5.2f}%  "
                      f"(n {len(said_absent)}) -- the text is wrong when this "
                      f"is high")
            if len(said_present):
                print(f"  text said PRESENT, rung low: measured vegetation "
                      f"median {said_present.veg.median()*100:5.2f}%  "
                      f"(n {len(said_present)}) -- the rung is wrong when this "
                      f"is high")

    for q in PAIRS:
        col = q + "_contra"
        if col not in j.columns:
            continue
        ex = j[j[col]].head(args.examples)
        if not len(ex):
            continue
        print(f"\n--- {q}: worst disagreements ---")
        for _, x in ex.iterrows():
            print(f"  {x.file}   rung {x[q + '_rung']:.1f}"
                  + (f"   measured veg {x.veg*100:.2f}%"
                     if q == "greenery" and pd.notna(x.veg) else ""))
            print(f"    {str(x[q])[:170]}")


if __name__ == "__main__":
    main()
