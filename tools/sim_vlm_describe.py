"""Ask the model to say what it sees, in its own words.

The ratings are read from logits at a forced prefix -- the model never writes a
token, it only reveals how it would have distributed one. That is the right
instrument for a number: it cannot ramble, cannot refuse, cannot be talked into
a different answer by its own preamble, and it costs one forward pass.

It also means nobody has ever heard the model explain itself, which is a real
gap when the numbers go in front of people who reasonably ask what the model
was looking at. This asks in open text instead, on a sample rather than on
everything, and keeps the answers beside the ratings so the two can be read
together.

WHAT THIS IS NOT. It is not validation and must not be quoted as any. A model
asked to justify a rating will produce a fluent justification whether or not
that reasoning drove the rating -- the text is generated after the fact and by
a different decoding path than the logit read. Treat it as an illustration of
what the imagery contains, and keep the measured twins in
tools/validation_figure.py as the evidence.

SAMPLED ACROSS THE RANGE, not at random. A random draw of a hundred frames from
Murray Hill is a hundred mid-scoring streets; stratifying by M means the extremes
-- the ones anybody will ask about -- are actually in the sample.

    .venv-gpu/Scripts/python tools/sim_vlm_describe.py --src data/raw/svi_180 \
        --mast-set svi_90_wide --n 90
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
from common import RES, banner
from mast import erase_mast

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
NAME = re.compile(r"(\d+)_(n\d+)_([NESW])(?:_([LRF]))?\.jpg$")

SYSTEM = ("You are describing a street for an urban design study. Be concrete "
          "and specific about what is physically present. Do not speculate "
          "about the neighbourhood, the residents, or the city.")

# Open questions, deliberately not the rung prompts. Asking "why did you say 5"
# invites the model to reverse-engineer a justification; asking what is there
# gets a description that can be checked against the photograph.
QUESTIONS = {
    "scene": "In two sentences, what does this street look like to walk down?",
    "greenery": "What vegetation is visible, and where is it -- street trees, "
                "planting at ground level, greenery on the buildings, or none?",
    "ground": "Describe the ground plane: the footway, its width and surface, "
              "kerbs, and anything standing on it.",
    "frontage": "Describe the building frontage at street level: entrances, "
                "windows, shopfronts, or blank wall.",
    "standout": "What is the single most noticeable thing in this view?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--calc", type=Path, default=None,
                    help="vlm_calculations.csv, to stratify the sample by M")
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mast-set", default=None)
    ap.add_argument("--max-new", type=int, default=110)
    ap.add_argument("--fields", nargs="+", default=list(QUESTIONS))
    ap.add_argument("--checkpoint", type=int, default=10)
    args = ap.parse_args()
    banner("what the model says it sees")

    files = sorted(args.src.rglob("*.jpg"))
    rows = []
    for p in files:
        m = NAME.search(p.name)
        if m:
            rows.append({"file": str(p.relative_to(args.src)).replace("\\", "/"),
                         "path": p, "node_id": m.group(2)})
    fl = pd.DataFrame(rows)
    print(f"{len(fl)} frames in {args.src}")

    # stratify by M so the sample spans the range rather than clustering at the
    # median, where a random draw would put nearly all of it
    calc = args.calc or (RES / "tables" / "vlm_calculations.csv")
    if calc.exists():
        c = pd.read_csv(calc)
        mcol = "M_noA" if "M_noA" in c.columns else "M"
        per = c.groupby("node_id")[mcol].mean().rename("M").reset_index()
        fl = fl.merge(per, on="node_id", how="left")
        ok = fl.M.notna()
        if ok.any():
            fl["band"] = pd.qcut(fl.loc[ok, "M"], 5, labels=False,
                                 duplicates="drop")
            per_band = max(1, args.n // int(fl.band.nunique()))
            fl = (fl[ok].groupby("band", group_keys=False)
                  .apply(lambda g: g.sample(min(per_band, len(g)),
                                            random_state=args.seed)))
            print(f"sampled {len(fl)} frames, stratified into "
                  f"{int(fl.band.nunique())} bands of M "
                  f"({fl.M.min():.3f} to {fl.M.max():.3f})")
    if len(fl) > args.n:
        fl = fl.sample(args.n, random_state=args.seed)
    if "M" not in fl.columns:
        print(f"no {calc.name}: sampling at random instead of by M")

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

    mast_set = args.mast_set or args.src.name
    if str(mast_set).lower() == "none":
        mast_set = None
        print("mast erase: DISABLED")
    else:
        print(f"mast erase: calibration set {mast_set!r}")

    out, ask = [], [(k, QUESTIONS[k]) for k in args.fields if k in QUESTIONS]
    for i, r in enumerate(tqdm(list(fl.itertuples()), desc="frames",
                               mininterval=10.0), 1):
        im = Image.open(r.path).convert("RGB")
        if mast_set:
            im, _ = erase_mast(im, mast_set)
        rec = {"file": r.file, "node_id": r.node_id,
               "M": getattr(r, "M", np.nan)}
        for key, q in ask:
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new,
                                     do_sample=False)
            reply = proc.batch_decode(
                gen[:, inp["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip().replace("\n", " ")
            rec[key] = reply
        out.append(rec)
        if i % args.checkpoint == 0:
            t = args.table or (RES / "tables" / "vlm_descriptions.csv")
            t.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(out).to_csv(t, index=False)

    t = args.table or (RES / "tables" / "vlm_descriptions.csv")
    d = pd.DataFrame(out)
    d.to_csv(t, index=False)
    print(f"\nwrote {t}  ({len(d)} frames x {len(ask)} questions)")
    if "M" in d.columns and d.M.notna().any():
        lo = d.nsmallest(1, "M").iloc[0]
        hi = d.nlargest(1, "M").iloc[0]
        for lab, r in (("lowest M", lo), ("highest M", hi)):
            print(f"\n  {lab} = {r.M:.3f}   {r.file}")
            for key, _ in ask[:3]:
                print(f"    {key:<10}{str(r[key])[:150]}")


if __name__ == "__main__":
    main()
