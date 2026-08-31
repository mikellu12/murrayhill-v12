"""SIM rated on three 60-degree cones, weighted as the perceptual literature.

The 1440 px panorama covers 180 degrees at exactly 8 px per degree, so the
cones are clean crops at 0-480, 480-960 and 960-1440 -- left, centre, right,
each 60 degrees, centred on the street axis in the direction of travel.

WHY THE WEIGHTS ARE IMPOSED RATHER THAN LEFT TO THE MODEL. The obvious worry
is double-counting: if the VLM already attends to the centre the way a person
does, weighting again in code applies the same correction twice. Measured on
102 images against the whole-panorama ratings, it does not. Implied centre
weight 0.36 by LMG and 0.36 by non-negative least squares, against the
literature's 0.80, and uniform thirds beat every alternative on out-of-sample
R2 with faces held out. The model reads a 180-degree view flat, so the
perceptual weighting has to come from here.

Every field is a rating. No counts: the manuscript's SIM terms are all
proportions or indices in [0,1], so a tally has the wrong type, and the
weighting then applies uniformly with no special case for fractional objects.
street_trees and people were dropped for that reason -- trees are already
carried by vertical_greenery, and people belong to t_base in the dwell
equation, not to the index.

rest_affordance pools what were three near-empty count fields. seating_places
came back zero on 99.4 per cent of images and planters on 98.8; sim_dwell.py
documents the same problem in the pixel data and solves it the same way, by
pooling seating, shelter and soft buffers into one presence signal.

Weighting is arithmetic here, never an instruction. Telling a model to weight
its own attention invites it to do the arithmetic badly and untraceably; the
weights live in config.yaml as declared parameters instead, and every cone's
raw rating is kept so the composite can be recomputed under other weights.

    .venv-gpu/Scripts/python tools/svi_180_cone_sim.py --show-prompt
    .venv-gpu/Scripts/python tools/svi_180_cone_sim.py --sample 12
    .venv-gpu/Scripts/python tools/svi_180_cone_sim.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, weights

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
CONES = ("left", "centre", "right")
# Declared, not fitted, like every other weight in this study. Asserted to sum
# to 1 before use: weights that do not sum to 1 silently push the index out of
# [0,1] and nothing downstream would complain.
WEIGHTS = {"left": 0.10, "centre": 0.80, "right": 0.10}

SYSTEM = ("You are an expert urban morphologist evaluating streetscape "
          "quality at eye-level (1.5m).")

CONE_LINE = ("This is the {} 60-degree third of a pedestrian's forward view, "
             "looking along the street in the direction of travel.\n\n")

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
  "rest_affordance": <1-7>,
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
rest_affordance       somewhere to sit or lean: benches, ledges, stoops,
                      steps, planter edges

greenery_layer is the dominant vegetation layer, or "none" when there is none.
dominant_edge is what the walking surface mostly runs alongside.

Output JSON only."""

RATE = ["vertical_greenery", "vertical_hardscape", "green_eye_level",
        "green_softening", "signage_detail", "enclosure", "facade_variation",
        "walking_room", "ground_floor_activity", "rest_affordance"]
CAT = ["greenery_layer", "dominant_edge"]
NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")


def prompt_for(cone):
    return CONE_LINE.format(cone.upper()) + SCHEMA


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
    out = {k: _num(j.get(k)) for k in RATE}
    for k in CAT:
        v = j.get(k)
        out[k] = str(v).strip().lower() if v is not None else None
    return out


def combine(d):
    """Weighted composite per field, plus the SIM dimensions."""
    w = WEIGHTS
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"cone weights must sum to 1: {w}"
    n = lambda s: (s.clip(1, 7) - 1) / 6.0
    for f in RATE:
        cols = [f"{c}_{f}" for c in CONES]
        if all(c in d.columns for c in cols):
            d[f] = sum(w[c] * d[f"{c}_{f}"] for c in CONES)
            d[f + "_n"] = n(d[f])

    W = CFG["sim_vlm"]["weights"]
    a, b, g, dw = (weights(W["imageability"]), weights(W["identity"]),
                   weights(W["dependence"]), weights(W["dimension"]))
    vg, vh = d.vertical_greenery.clip(1, 7), d.vertical_hardscape.clip(1, 7)
    d["nat_built"] = vg / (vg + vh)
    d["Imageability"] = (a["nat_built"] * d.nat_built
                         + a["gvi_eye"] * d.green_eye_level_n
                         + a["gmi"] * d.green_softening_n)
    d["Identity"] = (b["signboard"] * d.signage_detail_n
                     + b["enclosure"] * d.enclosure_n
                     + b["sfv"] * d.facade_variation_n)
    d["Dependence"] = (g["sidewalk_paver"] * d.walking_room_n
                       + g["sfv"] * d.facade_variation_n
                       + g["gfapi"] * d.ground_floor_activity_n)
    d["SIM_cone"] = (dw["imageability"] * d.Imageability
                     + dw["identity"] * d.Identity
                     + dw["dependence"] * d.Dependence)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_cone_sim.csv")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print exactly what the model receives, then stop")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    if args.show_prompt:
        banner("the prompt, exactly as the model receives it")
        print(f"weights: {WEIGHTS}   sum {sum(WEIGHTS.values()):.2f}")
        print(f"cones: 1440 px / 180 deg = 8 px per degree -> "
              f"0-480 | 480-960 | 960-1440\n")
        print("-" * 72)
        print("SYSTEM\n" + SYSTEM)
        for c in CONES:
            print("\n" + "-" * 72)
            print(f"USER  [image = {c} crop, 480x916]\n")
            print(prompt_for(c))
        return

    banner("SIM on three 60-degree cones")
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)

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
        fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in fl.groupby("street")]).head(args.sample)
    if fl.empty:
        print("nothing to do")
        return
    print(f"{len(fl)} images x 3 cones = {len(fl) * 3} calls")

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    out = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
        im = Image.open(r.path).convert("RGB")
        W, H = im.size
        third = W // 3
        rec = {"file": r.file, "street": r.street, "direction": r.direction,
               "node_id": r.node_id, "cardinal": r.cardinal, "seq": r.seq}
        for i, cone in enumerate(CONES):
            crop = im.crop((i * third, 0, (i + 1) * third if i < 2 else W, H))
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text",
                                                  "text": prompt_for(cone)}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=[text], images=[crop], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=proc.tokenizer.eos_token_id)
            got = parse(proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True))
            for k, v in got.items():
                rec[f"{cone}_{k}"] = v
        out.append(rec)
        if len(out) % 25 == 0:
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.table, index=False)

    d = combine(pd.concat([done, pd.DataFrame(out)], ignore_index=True))
    d = d.sort_values(["street", "direction", "seq"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\n{len(d)} rows -> {args.table}")
    for c in ["Imageability", "Identity", "Dependence", "SIM_cone"]:
        if c in d:
            s = d[c].dropna()
            print(f"  {c:<14} mean {s.mean():.3f}  sd {s.std():.3f}  "
                  f"min {s.min():.3f}  max {s.max():.3f}")


if __name__ == "__main__":
    main()
