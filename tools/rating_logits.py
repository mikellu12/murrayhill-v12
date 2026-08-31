"""Read the model's distribution over 1-7 instead of its argmax.

No field ever answers 4. Across 24,516 calls, on all nine fields, the exact
midpoint never appears. Two explanations stack:

  GREEDY DECODING HIDES IT. do_sample=False takes the argmax. If a frame sits
  at P(3)=.35 P(4)=.33 P(5)=.30, the model is saying "about the middle" and
  the decoder writes 3. A value that is never the single most likely token
  never appears, however much mass it carries -- so the gap may be an artefact
  of how the answer is read, not of what the model believes.

  THE MIDPOINT HAS NO WORDS. Each prompt names 1 and 7 and nothing between,
  so the model interpolates between two verbal anchors and lands on the
  neighbours of a point it was never given language for.

This forces the answer prefix, runs one forward pass, and reads the
next-token probabilities for the digits 1-7 directly. Two quantities come
out: the argmax, which is what the run recorded, and the expected value
sum(p_k * k), which is continuous, uses the whole distribution, and cannot
have a hole at 4.

Scored against the vegetation share measured over the same 90 degrees, so
"better" means tracks the street better, not "looks smoother".

    .venv-gpu/Scripts/python tools/rating_logits.py --n 64
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
from sim_fields import SYSTEM, prompt
from sim_vlm_run import NAME_90

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
FIELDS = ["green_eye_level", "vertical_greenery", "sky_openness",
          "facade_variation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    banner("the distribution behind the answer")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_90.search(jpg.name)
        if m:
            rows.append({"src_path": jpg, "node_id": m.group(2),
                         "side": m.group(4),
                         "file": str(jpg.relative_to(args.src)).replace("\\", "/")})
    fl = pd.DataFrame(rows)
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI"]]
    fl = fl.merge(met, on="node_id", how="left").dropna(subset=["GVI"])
    fl["q"] = pd.qcut(fl.GVI, 4, labels=False, duplicates="drop")
    per = max(1, args.n // fl.q.nunique())
    fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                    for _, g in fl.groupby("q")]).head(args.n).reset_index(drop=True)
    print(f"{len(fl)} half-views x {len(FIELDS)} fields\n")

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
    tok = proc.tokenizer

    # the digit tokens, however this tokenizer spells them
    ids = {}
    for k in range(1, 8):
        cand = [tok.encode(str(k), add_special_tokens=False),
                tok.encode(" " + str(k), add_special_tokens=False)]
        ids[k] = [c[0] for c in cand if len(c) == 1]
        if not ids[k]:
            sys.exit(f"digit {k} is not a single token")
    print("digit token ids:", {k: v for k, v in ids.items()}, "\n")

    out = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
        im = Image.open(r.src_path).convert("RGB")
        rec = {"file": r.file, "node_id": r.node_id, "side": r.side}
        for f in FIELDS:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text",
                                                  "text": prompt(f)}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            # force the answer prefix so the very next token is the digit
            text += f'{{"{f}": '
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inp).logits[0, -1].float()
            p = torch.softmax(logits, -1)
            mass = np.array([float(sum(p[i] for i in ids[k])) for k in range(1, 8)])
            mass = mass / mass.sum()          # renormalise over the 7 digits
            rec[f + "_argmax"] = int(np.argmax(mass) + 1)
            rec[f + "_ev"] = float((mass * np.arange(1, 8)).sum())
            rec[f + "_p4"] = float(mass[3])
            rec[f + "_top"] = float(mass.max())
        out.append(rec)
    d = pd.DataFrame(out)
    p = RES / "tables" / "rating_logits.csv"
    d.to_csv(p, index=False)

    print(f"\n  {'field':<20}{'P(4) mean':>11}{'P(4) max':>10}"
          f"{'argmax=4':>10}{'top-p mean':>12}")
    for f in FIELDS:
        print(f"  {f:<20}{d[f+'_p4'].mean():>11.3f}{d[f+'_p4'].max():>10.3f}"
              f"{int((d[f+'_argmax'] == 4).sum()):>10}{d[f+'_top'].mean():>12.3f}")

    print(f"\n  {'field':<20}{'argmax distinct':>16}{'EV min':>9}{'EV max':>9}"
          f"{'EV sd':>8}")
    for f in FIELDS:
        print(f"  {f:<20}{d[f+'_argmax'].nunique():>16}"
              f"{d[f+'_ev'].min():>9.2f}{d[f+'_ev'].max():>9.2f}"
              f"{d[f+'_ev'].std():>8.2f}")
    print(f"\nwrote {p}")
    print("  score it against the measured arc with tools/rating_logits_score.py")


if __name__ == "__main__":
    main()
