"""Is the twelve-field schema what is flattening the SIM ratings?

The 90-degree run collapsed to the rails -- vertical_greenery 1 on 90 per cent
of images, enclosure 7 on 89 per cent, and greenery correlating with measured
GVI at rho +0.28 where the same model on a short prompt reached +0.71. The
suspect is prompt length, because the same effect was measured once before on
enclosure alone: +0.012 asked inside the schema, +0.480 asked on its own.

This asks the same images the same questions three ways and scores all three
against measured GVI and VEI:

  full    the twelve-field schema exactly as svi_90_sim.py sends it
  group   the ten ratings split into three themed calls, each field anchored
  solo    one field per call, phrased like the prompt that scored +0.705

Anchoring is confounded with length on purpose. The schema states the scale
once for all ten fields ("none/absent at 1 to dominant/continuous at 7") while
the validated short prompt anchors both ends of that specific field. If short
prompts win, the follow-up question is which half did it; if they do not, the
schema is exonerated and the problem is elsewhere.

Sampled stratified by measured GVI so the range is present. A sample drawn at
random from Murray Hill is mostly bare, and a flat rating looks correct on bare
frontage -- the leafy tail is where a degenerate model gives itself away.

    .venv-gpu/Scripts/python tools/prompt_probe.py --n 48
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
from common import PROC, RES, banner

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])_([LR])\.jpg$")
WHOLE_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")
SIDE_NAME = {"L": "LEFT", "R": "RIGHT"}

SYSTEM = ("You are an expert urban morphologist evaluating streetscape "
          "quality at eye-level (1.5m).")
SIDE_LINE = ("This is the {} 90-degree half of the forward view along the "
             "street, in the direction of travel.\n\n")

# ---------------------------------------------------------------- variant 1
FULL_SCHEMA = """Judge this Manhattan streetscape. Reply with ONE JSON object, nothing else.

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

# ------------------------------------------------- anchors, one per field
# Both ends written out for every field, the way the +0.705 prompt did it.
ANCHOR = {
 "vertical_greenery":  ("no canopy or green facade at all",
                        "canopy and green facade dominating the view"),
 "vertical_hardscape": ("almost no built vertical surface in view",
                        "built wall, glazing and columns filling the view"),
 "green_eye_level":    ("no greenery at or below eye height",
                        "greenery at eye height along the whole frontage"),
 "green_softening":    ("greenery does nothing to relieve the enclosure",
                        "greenery substantially relieves the enclosure"),
 "signage_detail":     ("blank, no signage, cornices or mouldings",
                        "dense signage, cornices and shopfront lettering"),
 "enclosure":          ("wide open, sky dominating",
                        "a deep canyon where the sky is a slot"),
 "facade_variation":   ("one uniform frontage the whole way",
                        "the frontage changes every few metres"),
 "walking_room":       ("almost no walking surface on this side",
                        "a wide generous walking surface on this side"),
 "ground_floor_activity": ("continuous blank wall at street level",
                           "continuous active glazed shopfronts"),
 "rest_affordance":    ("nowhere at all to sit or lean",
                        "many places to sit or lean: benches, ledges, stoops"),
}

GROUPS = {
    "green": ["vertical_greenery", "vertical_hardscape", "green_eye_level",
              "green_softening"],
    "form":  ["signage_detail", "enclosure", "facade_variation"],
    "ground": ["walking_room", "ground_floor_activity", "rest_affordance"],
}

RATE = list(ANCHOR)
# what each rating is supposed to track, and in which direction
TARGETS = [("vertical_greenery", "GVI", +1), ("green_eye_level", "GVI", +1),
           ("green_softening", "GVI", +1), ("enclosure", "VEI", +1),
           ("vertical_hardscape", "VEI", +1)]


def prompt_group(fields):
    lines = "\n".join(f'  "{f}": <1-7>,' for f in fields).rstrip(",")
    body = "\n".join(f"{f}\n    1 = {ANCHOR[f][0]}\n    7 = {ANCHOR[f][1]}"
                     for f in fields)
    return (f"Judge this Manhattan streetscape. Reply with ONE JSON object, "
            f"nothing else.\n\n{{\n{lines}\n}}\n\n{body}\n\nOutput JSON only.")


def prompt_solo(field):
    lo, hi = ANCHOR[field]
    return (f"Rate this Manhattan street view. Reply with ONE JSON object and "
            f"nothing else: {{\"{field}\": <1-7>}} where 1 is {lo} and 7 is "
            f"{hi}. Use the whole 1-7 range.")


def parse(txt, fields):
    m = re.search(r"\{.*\}", txt, re.S)
    out = {f: np.nan for f in fields}
    if not m:
        return out
    try:
        j = json.loads(m.group())
    except json.JSONDecodeError:
        return out
    for f in fields:
        v = j.get(f)
        if isinstance(v, dict):
            v = v.get("score", v.get("value"))
        try:
            out[f] = float(v)
        except (TypeError, ValueError):
            g = re.search(r"-?\d+(\.\d+)?", str(v))
            out[f] = float(g.group()) if g else np.nan
    return out


def spearman_ci(x, y, groups, n=2000, rng=np.random.default_rng(0)):
    s = pd.DataFrame({"x": x, "y": y, "g": groups}).dropna()
    if len(s) < 12 or s.x.nunique() < 2:
        return None
    r = s.x.corr(s.y, method="spearman")
    uniq = pd.unique(s.g)
    idx = {q: np.flatnonzero(s.g.to_numpy() == q) for q in uniq}
    bs = []
    for _ in range(n):
        sub = s.iloc[np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])]
        if sub.x.nunique() > 1:
            bs.append(sub.x.corr(sub.y, method="spearman"))
    if not bs:
        return None
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(s)


def sample(src, n, seed, nodes=None, on="GVI"):
    """Half-views if the names carry a side, whole forward views if not."""
    rows = []
    for jpg in sorted(Path(src).rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        side = m.group(4) if m else None
        if not m:
            m = WHOLE_RE.search(jpg.name)
        if m:
            rows.append({"path": jpg, "node_id": m.group(2), "side": side or "W",
                         "file": str(jpg.relative_to(src)).replace("\\", "/")})
    fl = pd.DataFrame(rows)
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI", "face_id"]]
    fl = fl.merge(met, on="node_id", how="left").dropna(subset=["GVI"])
    if nodes is not None:
        fl = fl[fl.node_id.isin(set(nodes))]
        return (fl.groupby("node_id", as_index=False).first()
                  .head(n).reset_index(drop=True))
    fl = fl.dropna(subset=[on])
    fl["q"] = pd.qcut(fl[on], 4, labels=False, duplicates="drop")
    per = max(1, n // fl.q.nunique())
    out = pd.concat([g.sample(min(len(g), per), random_state=seed)
                     for _, g in fl.groupby("q")])
    return out.head(n).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "prompt_probe.csv")
    ap.add_argument("--match-nodes", type=Path, default=None,
                    help="reuse the node set from an earlier probe table")
    ap.add_argument("--variants", nargs="+", default=["full", "group", "solo"])
    ap.add_argument("--stratify", default="GVI", choices=["GVI", "VEI"],
                    help="which measured quantity the sample must spread")
    ap.add_argument("--no-side", action="store_true",
                    help="drop the 'this is the LEFT half' preamble, so the "
                         "text matches the whole-view prompt byte for byte")
    args = ap.parse_args()
    banner("does the schema flatten the ratings?")

    nodes = None
    if args.match_nodes and args.match_nodes.exists():
        nodes = pd.read_csv(args.match_nodes).node_id.unique()
        print(f"matching the {len(nodes)} nodes from {args.match_nodes.name}")
    fl = sample(args.src, args.n, args.seed, nodes, args.stratify)
    print(f"stratified on {args.stratify}: "
          f"{fl[args.stratify].min():.2f} to {fl[args.stratify].max():.2f}")
    print(f"{len(fl)} views, GVI {fl.GVI.min():.1f} to {fl.GVI.max():.1f}, "
          f"{fl.node_id.nunique()} nodes, sides {dict(fl.side.value_counts())}")
    per = (("full" in args.variants) + len(GROUPS) * ("group" in args.variants)
           + 3 * ("solo" in args.variants))
    calls = len(fl) * per
    print(f"{calls} model calls\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    def ask(im, text, max_new):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": text}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=proc.tokenizer.eos_token_id)
        return proc.tokenizer.decode(g[0][inp.input_ids.shape[1]:],
                                     skip_special_tokens=True)

    SOLO = ["vertical_greenery", "green_eye_level", "enclosure"]
    recs = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
        im = Image.open(r.path).convert("RGB")
        head = ("" if args.no_side or r.side not in SIDE_NAME
                else SIDE_LINE.format(SIDE_NAME[r.side]))
        base = {"file": r.file, "node_id": r.node_id, "side": r.side,
                "GVI": r.GVI, "VEI": r.VEI, "face_id": r.face_id}

        if "full" in args.variants:
            got = parse(ask(im, head + FULL_SCHEMA, 240), RATE)
            recs.append({**base, "variant": "full", **got})

        if "group" in args.variants:
            merged = {}
            for fields in GROUPS.values():
                merged.update(parse(ask(im, head + prompt_group(fields), 120), fields))
            recs.append({**base, "variant": "group", **merged})

        if "solo" in args.variants:
            solo = {}
            for f in SOLO:
                solo.update(parse(ask(im, head + prompt_solo(f), 24), [f]))
            recs.append({**base, "variant": "solo", **solo})

    d = pd.DataFrame(recs)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)

    print(f"\n=== Spearman vs measured, {len(fl)} half-views, "
          f"bootstrap clustered on face ===")
    print(f"  {'field':<22}{'variant':<8}{'rho':>7}   {'95% CI':<18}"
          f"{'mode%':>7}{'distinct':>9}")
    rows = []
    for field, tgt, _ in TARGETS:
        for v in args.variants:
            s = d[d.variant == v]
            if field not in s or s[field].dropna().empty:
                continue
            got = spearman_ci(s[field].to_numpy(float), s[tgt].to_numpy(float),
                              s.face_id.fillna(s.node_id).to_numpy())
            if not got:
                continue
            rho, lo, hi, n = got
            vals = s[field].dropna()
            mode_pct = (vals == vals.mode().iloc[0]).mean() * 100
            print(f"  {field:<22}{v:<8}{rho:>+7.3f}   "
                  f"[{lo:+.3f},{hi:+.3f}]  {mode_pct:>6.0f}%{vals.nunique():>9}")
            rows.append({"field": field, "target": tgt, "variant": v, "rho": rho,
                         "lo": lo, "hi": hi, "n": n, "mode_pct": mode_pct,
                         "distinct": vals.nunique()})
        print()
    pd.DataFrame(rows).to_csv(
        args.table.with_name(args.table.stem + "_summary.csv"), index=False)

    print("  L vs R separation (identical means = the model is not "
          "distinguishing sides)")
    for v in args.variants:
        s = d[d.variant == v]
        bits = []
        for f in ("vertical_greenery", "enclosure"):
            if f in s and s[f].notna().any():
                g = s.groupby("side")[f].mean()
                bits.append(f"{f} L {g.get('L', np.nan):.2f} R {g.get('R', np.nan):.2f}")
        print(f"    {v:<8}" + "   ".join(bits))
    print(f"\nwrote {args.table}")


if __name__ == "__main__":
    main()
