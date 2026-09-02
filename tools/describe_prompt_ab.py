"""Does enumerating the options make the model invent them?

The greenery question hands the model a menu -- "street trees, planting at
ground level, greenery on the buildings, or none" -- and at n00501 the answer
came back with two items off that menu, one of them false: greenery on the
buildings, on a side where segmentation measures 0.2 per cent vegetation. The
suspicion is that listing the categories invites the model to populate them.

That is a claim about the prompt, so it is settled by changing the prompt and
counting, not by reasoning about it. Two wordings, same frames, same decoding:

  MENU   the question as asked in the run, categories enumerated
  OPEN   the same question with the list removed

SCORED AGAINST SEGMENTATION, not against a reading of the answers. Each frame
carries a measured vegetation share, so an answer that asserts vegetation on a
frame with none is a false assertion and an answer that denies it on a frame
full of trees is a miss. Both matter: the menu could be removed and the model
could simply stop reporting real greenery, which would be worse.

FRAMES ARE STRATIFIED, half bare and half vegetated. Sampling at random from
Murray Hill would give mostly middling frames, where neither error shows.

WHAT THIS CANNOT SETTLE. It compares two wordings on a sample; it does not say
the winning wording is correct, only that it asserts less of what is not there.
The descriptions remain illustration and not validation either way.

    .venv-gpu/Scripts/python tools/describe_prompt_ab.py --n 30
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
from mast import erase_mast
from sim_vlm_describe import MODEL, MAX_PIXELS, SYSTEM

# The wording used in the run, and the same question with the list taken out.
# Only the enumeration differs; the system prompt and decoding are identical,
# so anything that moves is attributable to the menu.
PROMPTS = {
    "menu": ("What vegetation is visible and where -- street trees, planting "
             "at ground level, greenery on the buildings, or none?"),
    "open": "What vegetation is visible in this view, and where?",
}

# An answer denies vegetation if it says so. Anything else is read as asserting
# some, which is deliberately the generous reading for the menu wording.
DENY = re.compile(r"\b(no|not|none|without|absent|lacks?)\b[^.]{0,40}"
                  r"\b(vegetation|greenery|plants?|trees?|planting)\b"
                  r"|\bno visible\b|\bnone\b", re.I)
BUILDING = re.compile(r"\bgreener\w*\b[^.]{0,30}\bbuilding", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--seg", type=Path,
                    default=PROC / "seg90_two_model.csv")
    ap.add_argument("--n", type=int, default=30, help="frames, half of each")
    ap.add_argument("--bare", type=float, default=0.01)
    ap.add_argument("--green", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mast-set", default="svi_180")
    ap.add_argument("--max-new", type=int, default=110)
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "describe_prompt_ab.csv")
    args = ap.parse_args()
    banner("does the menu invent the greenery?")

    # measured vegetation per 180 frame: both 90 halves of that node averaged,
    # which is the same quantity the strip covers
    seg = pd.read_csv(args.seg)
    seg["node_id"] = seg.file.astype(str).str.extract(r"(n\d+)")[0]
    veg = seg.groupby("node_id").map_Vegetation.mean().rename("veg")

    rows = []
    for p in sorted(args.src.rglob("*.jpg")):
        m = re.search(r"(n\d+)", p.name)
        if m:
            rows.append({"path": p, "node_id": m.group(1),
                         "file": str(p.relative_to(args.src)).replace("\\", "/")})
    fl = pd.DataFrame(rows).merge(veg, on="node_id", how="inner")
    print(f"{len(fl)} frames carry a measured vegetation share")

    k = max(1, args.n // 2)
    bare = fl[fl.veg < args.bare].sample(min(k, (fl.veg < args.bare).sum()),
                                         random_state=args.seed)
    lush = fl[fl.veg > args.green].sample(min(k, (fl.veg > args.green).sum()),
                                          random_state=args.seed)
    bare["kind"], lush["kind"] = "bare", "vegetated"
    fl = pd.concat([bare, lush], ignore_index=True)
    print(f"sampled {len(bare)} bare (<{args.bare:.0%} vegetation) and "
          f"{len(lush)} vegetated (>{args.green:.0%})\n")

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

    out = []
    for r in tqdm(list(fl.itertuples()), desc="frames", mininterval=10.0):
        im = Image.open(r.path).convert("RGB")
        if args.mast_set:
            im, _ = erase_mast(im, args.mast_set)
        rec = {"file": r.file, "node_id": r.node_id, "veg": r.veg,
               "kind": r.kind}
        for tag, q in PROMPTS.items():
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new,
                                     do_sample=False)
            a = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
            rec[tag] = a.replace("\n", " ")
            rec[tag + "_asserts"] = not bool(DENY.search(a))
            rec[tag + "_building"] = bool(BUILDING.search(a))
        out.append(rec)

    d = pd.DataFrame(out)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\nwrote {args.table}\n")

    b = d[d.kind == "bare"]
    v = d[d.kind == "vegetated"]
    print(f"{'':<10}{'false assertion':>18}{'building claim':>17}{'miss':>10}")
    print(f"{'':<10}{'(bare, n=' + str(len(b)) + ')':>18}"
          f"{'(bare)':>17}{'(veg, n=' + str(len(v)) + ')':>10}")
    for tag in PROMPTS:
        fa = b[tag + "_asserts"].mean() if len(b) else np.nan
        bc = b[tag + "_building"].mean() if len(b) else np.nan
        ms = (~v[tag + "_asserts"]).mean() if len(v) else np.nan
        print(f"{tag:<10}{fa:>17.0%}{bc:>17.0%}{ms:>10.0%}")
    print("\nlower is better on all three: the open wording wins only if it "
          "asserts less\nof what is not there WITHOUT missing what is.")


if __name__ == "__main__":
    main()
