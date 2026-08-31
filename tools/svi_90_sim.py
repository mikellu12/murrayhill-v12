"""SIM rated on the 90-degree halves, one row per side of each walk.

Reads data/raw/svi_90, where every image is already one 90-degree half
rendered on its own axis -- left or right of the walk bearing, facing the
frontage on that side. One model call per image; nothing is cropped here.

NO WEIGHTING BETWEEN SIDES. The left and right halves are not two views of one
thing to be averaged. They are two different sidewalks, and a pedestrian walks
one of them. Each (node, walk, side) is its own observation with its own SIM,
so a street carries four per node: two walks by two sides. Any street-level
figure is a plain mean over observations.

The 60/60/60 split with 0.80/0.10/0.10 was tried first and dropped. It models
a pedestrian's frontal attention, but Street View is captured from a vehicle
on the roadway, so that geometry does not apply. Measured separately, the
model does not centre-weight a 180-degree view either: implied centre weight
0.36 by LMG and 0.36 by non-negative least squares against a literature value
of 0.80, with uniform thirds beating every alternative on out-of-sample R2.

Every field is a rating. The manuscript's SIM terms are all proportions or
indices in [0,1], so counts have the wrong type; street_trees and people were
dropped for that reason, trees already being carried by vertical_greenery and
people belonging to t_base in the dwell equation rather than to the index.
rest_affordance pools what were three near-empty count fields -- seating_places
came back zero on 99.4 per cent of images and planters on 98.8 -- the same
pooling sim_dwell.py applies to the pixel data for the same reason.

SFV enters both Identity and Dependence, as the manuscript specifies. Note
that SVF here is SVF_band: the imagery spans only +/-45 degrees of elevation,
so the zenith is never sampled and this is not comparable to published SVF.

    .venv-gpu/Scripts/python tools/svi_90_sim.py --show-prompt
    .venv-gpu/Scripts/python tools/svi_90_sim.py --sample 12
    .venv-gpu/Scripts/python tools/svi_90_sim.py
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
SIDES = ("L", "R")
SIDE_NAME = {"L": "LEFT", "R": "RIGHT"}
# No weighting. The left and right halves are not two views of one thing to be
# combined -- they are two different sidewalks, and a pedestrian walks one of
# them. Each (node, walk, side) is its own observation with its own SIM, so a
# street carries four: two walks by two sides. Any street-level figure is a
# plain mean over observations, not a weighted field.

SYSTEM = ("You are an expert urban morphologist evaluating streetscape "
          "quality at eye-level (1.5m).")

SIDE_LINE = ("This is the {} 90-degree half of the forward view along the "
             "street, in the direction of travel. It faces the frontage on "
             "that side of the street.\n\n")
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
enclosure             how strongly this side's wall encloses the street
facade_variation      how much the frontage changes along the street
walking_room          how much room the walking surface gives on this side
ground_floor_activity active glazed shopfronts against blank wall, this side
rest_affordance       somewhere to sit or lean: benches, ledges, stoops,
                      steps, planter edges

greenery_layer is the dominant vegetation layer, or "none" when there is none.
dominant_edge is what the walking surface mostly runs alongside.

Output JSON only."""

RATE = ["vertical_greenery", "vertical_hardscape", "green_eye_level",
        "green_softening", "signage_detail", "enclosure", "facade_variation",
        "walking_room", "ground_floor_activity", "rest_affordance"]
CAT = ["greenery_layer", "dominant_edge"]
NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])_([LR])\.jpg$")


def prompt_for(side):
    return SIDE_LINE.format(SIDE_NAME[side]) + SCHEMA


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


def sim_terms(d):
    """SIM per observation. Each row is one side of one walk at one node."""
    n = lambda c: (d[c].clip(1, 7) - 1) / 6.0
    for f in RATE:
        d[f + "_n"] = n(f)

    W = CFG["sim_vlm"]["weights"]
    a, b, g, dw = (weights(W["imageability"]), weights(W["identity"]),
                   weights(W["dependence"]), weights(W["dimension"]))

    vg, vh = d.vertical_greenery.clip(1, 7), d.vertical_hardscape.clip(1, 7)
    d["nat_built"] = vg / (vg + vh)
    d["Imageability"] = (a["nat_built"] * d.nat_built
                         + a["gvi_eye"] * d.green_eye_level_n
                         + a["gmi"] * d.green_softening_n)
    # SFV enters Identity and Dependence both, as the manuscript specifies.
    d["Identity"] = (b["signboard"] * d.signage_detail_n
                     + b["enclosure"] * d.enclosure_n
                     + b["sfv"] * d.facade_variation_n)
    d["Dependence"] = (g["sidewalk_paver"] * d.walking_room_n
                       + g["sfv"] * d.facade_variation_n
                       + g["gfapi"] * d.ground_floor_activity_n)
    d["SIM"] = (dw["imageability"] * d.Imageability
                + dw["identity"] * d.Identity
                + dw["dependence"] * d.Dependence)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_90_sim.csv")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print exactly what the model receives, then stop")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    if args.show_prompt:
        banner("the prompt, exactly as the model receives it")
        print("no cone weighting: each side is its own observation")
        print("halves: rendered at FOV 90, 1440x1833, 16 px per degree,")
        print("        the forward 180 split at the street axis, left and right\n")
        print("-" * 72)
        print("SYSTEM\n" + SYSTEM)
        for c in SIDES:
            print("\n" + "-" * 72)
            print(f"USER  [image = {SIDE_NAME[c]} half, 1440x1833]\n")
            print(prompt_for(c))
        return

    banner("SIM on the 90-degree halves")
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
                         "cardinal": m.group(3), "side": m.group(4),
                         "seq": int(m.group(1))})
    fl = pd.DataFrame(rows)
    print(f"{len(fl)} half-views")

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
    print(f"{len(fl)} calls, one per half-view")

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    out = []
    for r in tqdm(list(fl.itertuples()), desc="half-views", mininterval=10.0):
        im = Image.open(r.path).convert("RGB")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": prompt_for(r.side)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=proc.tokenizer.eos_token_id)
        rec = parse(proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True))
        rec.update({"file": r.file, "street": r.street, "direction": r.direction,
                    "node_id": r.node_id, "cardinal": r.cardinal,
                    "side": r.side, "seq": r.seq})
        out.append(rec)
        if len(out) % 25 == 0:
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.table, index=False)

    d = sim_terms(pd.concat([done, pd.DataFrame(out)], ignore_index=True))
    d = d.sort_values(["street", "direction", "seq", "side"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\n{len(d)} rows -> {args.table}")
    for c in ["Imageability", "Identity", "Dependence", "SIM"]:
        if c in d:
            s = d[c].dropna()
            print(f"  {c:<14} mean {s.mean():.3f}  sd {s.std():.3f}  "
                  f"min {s.min():.3f}  max {s.max():.3f}")


if __name__ == "__main__":
    main()
