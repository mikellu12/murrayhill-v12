"""The manuscript's Street Interface Matrix, asked of a VLM as judgements.

    Imageability = a1*nat_built + a2*GVI_eye  + a3*GMI
    Identity     = b1*signage   + b2*enclosure + b3*SFV
    Dependence   = g1*walkroom  + g2*SFV       + g3*GFAPI
    SIM          = w_Img*Imageability + w_Id*Identity + w_Dep*Dependence

Weights are in config.yaml under `sim_vlm:`, declared rather than fitted.

WHY THIS ASKS FOR RATINGS AND NOT PERCENTAGES. The manuscript defines its
terms as pixel shares, P_x / P_total. Asked for those directly, this model
produces numbers that look fine and mean nothing: across 48 trial images not
one percentage field correlated with the measured quantity it names, every
confidence interval spanning zero. The same model rating eye-level greenery
1-7 reached rho +0.705 against measured GVI over 1252 images. Counting
discrete objects also works -- people counted against the segmenter's person
pixels, rho +0.683. Estimating what fraction of an image a class occupies is
the one thing segmentation is for and the one thing the VLM cannot do.

So every term is asked as an ordinal judgement of the THING, and discrete
objects are counted. No percentages.

THREE ASSEMBLIES, ONE RUN, so the manuscript's central claim can be measured
rather than asserted:

  SIM_vlm     every term from VLM ratings. What the paper claims.
  SIM_hybrid  VLM for the perceptual terms, measured pixels for the geometric
              ones -- enclosure from VEI, walking room from the segmenter's
              sidewalk share. What the evidence so far supports.
  sim_index.csv holds the ADE20K pixel index for the same nodes, joinable on
              node_id for a third comparison. Neither replaces the others.

Enclosure is the term to watch. Four VLM formulations of it have now failed
against measured geometry -- categorical, direct ratio, counted storeys, and
framing_score over the full 1254 -- so `enclosure` is expected to be the weak
column and is exactly what SIM_hybrid replaces.

Scale anchors describe the ENDS only. Naming a middle value teaches the model
to return it: given "4 = a few distinct buildings", facade variation came
back 4.000 on 17 of 17 images.

    .venv-gpu/Scripts/python tools/svi_180_sim_vlm.py --sample 68   # trial
    .venv-gpu/Scripts/python tools/svi_180_sim_vlm.py               # all 1254
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, weights

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
W = {k: weights(v) if isinstance(v, dict) else v
     for k, v in CFG["sim_vlm"]["weights"].items()}

SYSTEM = ("You are an expert urban morphologist evaluating streetscape "
          "quality at eye-level (1.5m).")

SCHEMA = """Judge this Manhattan streetscape. Reply with ONE JSON object, nothing else.

{
  "vertical_greenery": <1-7>,
  "vertical_hardscape": <1-7>,
  "green_eye_level": <1-7>,
  "green_softening": <1-7>,
  "signage_detail": <1-7>,
  "enclosure": <1-7>,
  "facade_variation": <1-7>,
  "walking_room": <1-7>,
  "ground_floor_activity": <1-7>,
  "street_trees": <integer>,
  "seating_places": <integer>,
  "planters": <integer>,
  "people": <integer>,
  "greenery_layer": "none|low_planter|hedge|tree_canopy|vertical_wall",
  "dominant_edge": "blank_wall|shopfront|stoop|railing|planting|construction"
}

Each 1-7 scale runs from none/absent at 1 to dominant/continuous at 7.

vertical_greenery     canopy, green facades, hedge walls above the ground
vertical_hardscape    built vertical surface: walls, glazing, columns
green_eye_level       greenery at or below a standing person's eye
green_softening       how far greenery relieves the enclosure
signage_detail        signage, cornices, mouldings, shopfront lettering
enclosure             how strongly the walls close the street into a room
facade_variation      how much the frontage changes along the street
walking_room          how much room the walking surface gives
ground_floor_activity active glazed shopfronts against blank wall

street_trees, seating_places, planters and people are counts. Zero is common.

greenery_layer is the dominant vegetation layer, or "none" when there is none.
dominant_edge is what the walking surface mostly runs alongside.

Output JSON only."""

NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")
RATE = ["vertical_greenery", "vertical_hardscape", "green_eye_level",
        "green_softening", "signage_detail", "enclosure", "facade_variation",
        "walking_room", "ground_floor_activity"]
COUNT = ["street_trees", "seating_places", "planters", "people"]
CAT = ["greenery_layer", "dominant_edge"]


def _num(v):
    if isinstance(v, dict):
        v = v.get("score", v.get("value"))
    try:
        return float(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(\.\d+)?", str(v))
        return float(m.group()) if m else np.nan


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        j = json.loads(m.group())
    except json.JSONDecodeError:
        return {}
    out = {k: _num(j.get(k)) for k in RATE + COUNT}
    for k in CAT:
        v = j.get(k)
        out[k] = str(v).strip().lower() if v is not None else None
    return out


def assemble(d):
    """The three dimensions, in a pure-VLM and a hybrid assembly."""
    n = lambda c: (d[c].clip(1, 7) - 1) / 6.0            # 1-7 -> [0,1]

    # A ratio of two ratings rather than of two pixel shares: bounded by
    # construction, so it cannot swamp the other terms the way an unbounded
    # ratio does -- the trap sim_dwell.py documents for H/W against SVF.
    vg, vh = d.vertical_greenery.clip(1, 7), d.vertical_hardscape.clip(1, 7)
    d["nat_built"] = vg / (vg + vh)
    d["gvi_eye_n"] = n("green_eye_level")
    d["gmi_n"] = n("green_softening")
    d["signage_n"] = n("signage_detail")
    d["enclosure_vlm"] = n("enclosure")
    d["sfv_n"] = n("facade_variation")
    d["walkroom_vlm"] = n("walking_room")
    d["gfapi_n"] = n("ground_floor_activity")

    a, b, g = W["imageability"], W["identity"], W["dependence"]
    dw = W["dimension"]

    def build(tag, enclosure, walkroom):
        d["Imageability_" + tag] = (a["nat_built"] * d.nat_built
                                    + a["gvi_eye"] * d.gvi_eye_n
                                    + a["gmi"] * d.gmi_n)
        d["Identity_" + tag] = (b["signboard"] * d.signage_n
                                + b["enclosure"] * enclosure
                                + b["sfv"] * d.sfv_n)
        d["Dependence_" + tag] = (g["sidewalk_paver"] * walkroom
                                  + g["sfv"] * d.sfv_n
                                  + g["gfapi"] * d.gfapi_n)
        d["SIM_" + tag] = (dw["imageability"] * d["Imageability_" + tag]
                           + dw["identity"] * d["Identity_" + tag]
                           + dw["dependence"] * d["Dependence_" + tag])

    build("vlm", d.enclosure_vlm, d.walkroom_vlm)

    # metrics.csv, NOT v12_metrics.csv. node_id is positional (CLAUDE.md):
    # the v12 frame's ids address different street corners, and joining to it
    # scores every judgement against the wrong location -- GVI correlates
    # +0.08 between the two files for the same id.
    #
    # Hybrid: the two terms the VLM has failed on come from measured pixels.
    # VEI is building/(building+sky), already a bounded enclosure measure;
    # the segmenter's sidewalk share stands in for walking room.
    ref = PROC / "metrics.csv"
    if ref.exists():
        v = pd.read_csv(ref)
        if "VEI" in v.columns and "VEI" not in d.columns:
            d = d.merge(v[["node_id", "VEI"]], on="node_id", how="left")
    seg = RES / "tables" / "svi_180_segformer.csv"
    if seg.exists():
        s = pd.read_csv(seg)
        if "sidewalk" in s.columns and "sidewalk" not in d.columns:
            d = d.merge(s[["file", "sidewalk"]], on="file", how="left")
    enc = d["VEI"] if "VEI" in d.columns else pd.Series(np.nan, index=d.index)
    if "sidewalk" in d.columns and d.sidewalk.notna().any():
        walk = d.sidewalk / d.sidewalk.max()
    else:
        walk = pd.Series(np.nan, index=d.index)
    build("hybrid", enc.fillna(d.enclosure_vlm), walk.fillna(d.walkroom_vlm))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_sim_vlm.csv")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("street interface matrix, vlm judgements")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            rows.append({"file": str(jpg.relative_to(args.src)).replace("\\", "/"),
                         "path": jpg, "street": jpg.parent.parent.name,
                         "direction": jpg.parent.name, "node_id": m.group(2),
                         "cardinal": m.group(3), "seq": int(m.group(1))})
    fl = pd.DataFrame(rows)
    print(f"{len(fl)} panoramas")

    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        fl = fl[~fl.file.isin(set(done.file))]
        print(f"{len(done)} already done, {len(fl)} remaining")
    if args.sample:
        per = max(1, args.sample // max(1, fl.street.nunique()))
        parts = [g.sample(min(len(g), per), random_state=args.seed)
                 for _, g in fl.groupby("street")]
        fl = pd.concat(parts).head(args.sample)
        print(f"trial sample: {len(fl)} across {fl.street.nunique()} streets")
    if fl.empty:
        print("nothing to do")
        return

    print(f"loading {MODEL} in 4-bit NF4")
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    out, bad = [], 0
    for r in tqdm(list(fl.itertuples()), desc="panoramas", mininterval=10.0):
        img = Image.open(r.path).convert("RGB")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": SCHEMA}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=proc.tokenizer.eos_token_id)
        rec = parse(proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True))
        if not rec:
            bad += 1
        rec.update({"file": r.file, "street": r.street, "direction": r.direction,
                    "node_id": r.node_id, "cardinal": r.cardinal, "seq": r.seq})
        out.append(rec)
        if len(out) % 25 == 0:
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.table, index=False)

    d = assemble(pd.concat([done, pd.DataFrame(out)], ignore_index=True))
    d = d.sort_values(["street", "direction", "seq"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\n{len(d)} rows -> {args.table}" + (f"   {bad} unparsed" if bad else ""))

    print("\n=== spread ===")
    for c in RATE + COUNT + ["nat_built", "Imageability_vlm", "Identity_vlm",
                             "Dependence_vlm", "SIM_vlm", "SIM_hybrid"]:
        if c in d and d[c].notna().any():
            s = d[c].dropna()
            print(f"  {c:<24} mean {s.mean():6.3f} sd {s.std():6.3f} "
                  f"min {s.min():6.2f} max {s.max():6.2f} distinct {s.nunique()}")
    for c in CAT:
        if c in d:
            print(f"  {c:<24} {d[c].value_counts().to_dict()}")

    # Scored against the pipeline's own measurements. A trial that only shows
    # spread cannot tell a good schema from a merely fluent one.
    ref = PROC / "metrics.csv"
    if ref.exists():
        v = pd.read_csv(ref)
        keep = [c for c in ("node_id", "GVI", "VEI", "SVF_band") if c in v.columns]
        j = d.merge(v[keep], on="node_id", how="inner", suffixes=("", "_m"))
        print(f"\n=== vs measured (n={len(j)}) ===")
        rng = np.random.default_rng(0)
        for a_, b_ in (("green_eye_level", "GVI"), ("vertical_greenery", "GVI"),
                       ("green_softening", "GVI"), ("Imageability_vlm", "GVI"),
                       ("enclosure", "VEI"), ("vertical_hardscape", "VEI"),
                       ("Identity_vlm", "VEI"), ("street_trees", "GVI")):
            bb = b_ if b_ in j.columns else b_ + "_m"
            if a_ not in j.columns or bb not in j.columns:
                continue
            s = j[[a_, bb]].dropna()
            if len(s) < 8 or s[a_].nunique() < 2:
                continue
            r = s[a_].corr(s[bb], method="spearman")
            bs = [s.iloc[rng.integers(0, len(s), len(s))].corr(
                method="spearman").iloc[0, 1] for _ in range(2000)]
            lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
            star = "  *" if lo > 0 or hi < 0 else ""
            print(f"  {a_:<22} vs {b_:<10} rho {r:+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]  n={len(s)}{star}")
        print("  * interval excludes zero")


if __name__ == "__main__":
    main()
