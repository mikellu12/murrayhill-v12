"""Can one sentence do what erasing pixels does?

Erasing Google's camera mast moves signage_detail by -0.104 of a rung
(Wilcoxon p<0.0001, n=47) and moves nothing else: the VLM reads the wordmark on
the mast as signage. Painting it out fixes that, but it puts a flat grey
rectangle into every frame, and "we inpainted a patch into all 3,064 images" is
a harder sentence to defend than "we told the model to ignore the camera".

So three arms on the same frames, differing in one thing each:

  raw       the frame as captured, prompt as it ships
  erased    mast painted out in the local median colour, prompt unchanged
  told      frame as captured, one line added to the system turn

`erased` is the reference: it is the ground truth for what removing the mast's
influence looks like, because the pixels are actually gone. `told` is only
useful if it lands near `erased` and far from `raw`.

Also rated: two fields the mast does NOT affect. If the added sentence moves
those, it is not removing a bias, it is just perturbing the model -- and a
prompt line that shifts unrelated answers is worse than the bias it fixes.

    .venv-gpu/Scripts/python tools/vlm_mast_prompt_probe.py --n 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner
from sim_fields import SYSTEM
from sim_scale import prompt7
from mast import erase_mast

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
# affected by the mast, and two controls that are not
FIELDS = ["signage_detail", "walkable_ground", "vertical_hardscape"]

TOLD = (" Every frame contains the camera vehicle's mast: a dark vertical post "
        "rising from the bottom edge with a Google watermark on it. It is "
        "photographic equipment, not part of the street. Ignore it entirely, "
        "including any text on it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=57)
    args = ap.parse_args()
    banner("prompt line vs erasing pixels")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    o = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    src = Path("data/raw/svi_90")
    o = o[[(src / f).exists() for f in o.file]]
    take = o.sample(args.n, random_state=args.seed)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=768 * 768)
    proc.tokenizer.padding_side = "left"
    net = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda").eval()
    tok = proc.tokenizer
    ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]
    ks = np.arange(1, 8)

    def rate(field, im, sys_extra=""):
        msgs = [{"role": "system", "content": SYSTEM + sys_extra},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt7(field)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t + '{"' + field + '": '], images=[im],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = net(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, ids], -1).cpu().numpy()
        return float(((p / p.sum()) * ks).sum())

    rows, skipped = [], 0
    for r in tqdm(list(take.itertuples()), desc="frames", mininterval=20.0):
        im = Image.open(src / r.file).convert("RGB")
        cln, m = erase_mast(im)
        if not m.any():
            skipped += 1
            continue
        rec = {"file": r.file}
        for f in FIELDS:
            rec[f + "_raw"] = rate(f, im)
            rec[f + "_erased"] = rate(f, cln)
            rec[f + "_told"] = rate(f, im, TOLD)
        rows.append(rec)

    d = pd.DataFrame(rows)
    out = RES / "tables" / "vlm_mast_prompt_probe.csv"
    d.to_csv(out, index=False)
    print(f"\n  n={len(d)} frames ({skipped} had no mast)\n")
    print(f"    {'field':<22}{'raw':>8}{'erased':>9}{'told':>8}"
          f"{'erased-raw':>12}{'told-raw':>10}")
    for f in FIELDS:
        a, b, c = d[f + "_raw"], d[f + "_erased"], d[f + "_told"]
        print(f"    {f:<22}{a.mean():>8.3f}{b.mean():>9.3f}{c.mean():>8.3f}"
              f"{(b-a).mean():>+12.3f}{(c-a).mean():>+10.3f}")
    print(f"\n  significance against raw (Wilcoxon):\n")
    print(f"    {'field':<22}{'erased':>10}{'told':>10}")
    for f in FIELDS:
        a, b, c = d[f + "_raw"], d[f + "_erased"], d[f + "_told"]
        pb = wilcoxon(a, b).pvalue if (a - b).abs().sum() > 0 else 1.0
        pc = wilcoxon(a, c).pvalue if (a - c).abs().sum() > 0 else 1.0
        print(f"    {f:<22}{pb:>10.4f}{pc:>10.4f}")
    print(f"\n  does the sentence reproduce the pixel edit?")
    print(f"    (correlation of the two shifts, and how much of it is recovered)\n")
    for f in FIELDS:
        a, b, c = d[f + "_raw"], d[f + "_erased"], d[f + "_told"]
        db, dc = b - a, c - a
        r = np.corrcoef(db, dc)[0, 1] if db.std() > 0 and dc.std() > 0 else np.nan
        frac = dc.mean() / db.mean() if abs(db.mean()) > 1e-9 else np.nan
        print(f"    {f:<22}shift corr {r:>+6.3f}   recovers {frac*100:>6.0f}%")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
