"""Two phrasings each for the three fields that stayed flat. Two minutes.

The smoke test fixed six of nine fields. Three did not move:

  walkable_ground     94% on one value -- WORSE than the walking_room wording
                      it replaced (73%), so the rewrite hurt
  facade_variation    81% on one value, 2 distinct, and it carries 22% of the
                      index, more than any other input
  vertical_hardscape  spread improved but rho against building share is +0.179

Only vertical_hardscape has a measured twin, so for the other two the test is
spread: a field answering one value on 80+ per cent of images cannot carry a
weight regardless of what it correlates with.

    .venv-gpu/Scripts/python tools/anchor_probe.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from sim_fields import SYSTEM
from sim_vlm_run import NAME_90, one

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28

# field -> variant -> (low, high).  "a" is what the smoke test used.
VARIANTS = {
    "walkable_ground": {
        "a": ("almost none of the ground surface is walkable, it is nearly all "
              "roadway",
              "the ground surface is almost entirely sidewalk and paving"),
        # back toward the phrasing that scored 73%, but naming the sidewalk
        # itself rather than how roomy it feels
        "b": ("the sidewalk is very narrow or missing on this side",
              "the sidewalk is very wide, taking up most of the ground in view"),
    },
    "facade_variation": {
        "a": ("one uniform frontage the whole way along",
              "the frontage changes every few metres"),
        # "changes every few metres" asks about a spatial sequence. Name the
        # things that change instead, which are visible in a single frame.
        "b": ("a single unbroken building face, one material, one window pattern",
              "many different building faces, materials, window patterns and "
              "entrances along the street"),
    },
    "vertical_hardscape": {
        "a": ("almost no built vertical surface in view",
              "building wall, glazing and columns filling the view"),
        # the measured twin is a share of frame, so ask for a share of frame
        "b": ("buildings take up almost none of this view",
              "buildings take up almost the entire view"),
    },
}
TWIN = {"vertical_hardscape": "arc_bld"}


def ask_text(field, lo, hi):
    return (f"Rate this Manhattan street view. Reply with ONE JSON object and "
            f"nothing else: {{\"{field}\": <1-7>}} where 1 is {lo} and 7 is "
            f"{hi}. Use the whole 1-7 range.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    banner("two phrasings each, three flat fields")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_90.search(jpg.name)
        if m:
            rows.append({"src_path": jpg, "node_id": m.group(2),
                         "side": m.group(4),
                         "file": str(jpg.relative_to(args.src)).replace("\\", "/")})
    fl = pd.DataFrame(rows)
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI"]]
    fl = fl.merge(met, on="node_id", how="left").dropna(subset=["VEI"])
    fl["q"] = pd.qcut(fl.VEI, 4, labels=False, duplicates="drop")
    per = max(1, args.n // fl.q.nunique())
    fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                    for _, g in fl.groupby("q")]).head(args.n).reset_index(drop=True)

    jobs = [(f, v, lo, hi) for f, vs in VARIANTS.items()
            for v, (lo, hi) in vs.items()]
    print(f"{len(fl)} images x {len(jobs)} prompts = {len(fl)*len(jobs)} calls\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    texts = []
    for f, v, lo, hi in jobs:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": ask_text(f, lo, hi)}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True))

    t0 = time.perf_counter()
    out = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=15.0):
        im = Image.open(r.src_path).convert("RGB")
        inp = proc(text=texts, images=[im] * len(jobs), padding=True,
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                               pad_token_id=proc.tokenizer.eos_token_id)
        cut = inp.input_ids.shape[1]
        rec = {"file": r.file, "node_id": r.node_id, "side": r.side}
        for (f, v, _, _), row in zip(jobs, g):
            rec[f"{f}__{v}"] = one(
                proc.tokenizer.decode(row[cut:], skip_special_tokens=True), f)
        out.append(rec)
    d = pd.DataFrame(out)
    print(f"\n  {time.perf_counter() - t0:.0f} s\n")

    print(f"  {'field':<20}{'variant':<9}{'mean':>6}{'sd':>6}{'mode%':>7}"
          f"{'distinct':>9}")
    for f in VARIANTS:
        for v in VARIANTS[f]:
            col = f"{f}__{v}"
            s = d[col].dropna()
            if s.empty:
                continue
            mo = s.mode().iloc[0]
            rho = ""
            tag = "  <-- flat" if (s == mo).mean() > 0.80 else ""
            print(f"  {f:<20}{v:<9}{s.mean():>6.2f}{s.std():>6.2f}"
                  f"{(s == mo).mean()*100:>6.0f}%{s.nunique():>9}{tag}")
        print()
    parts = d.file.str.split("/", expand=True)
    d["street"], d["walk"] = parts[0], parts[1]
    p = RES / "tables" / "anchor_probe.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False)
    print(f"wrote {p}")
    print("  score against the measured arc with:")
    print("    .venv/Scripts/python tools/anchor_score.py")


if __name__ == "__main__":
    main()
